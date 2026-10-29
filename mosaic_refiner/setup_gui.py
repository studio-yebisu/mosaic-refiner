"""Explicit first-run installation. Opening the window only reads metadata."""
from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog
import webbrowser

from .runtime_settings import load, RuntimeConnection, RUNTIME
from .setup_runtime import resources, resource_status, inspect_nodes
from .setup_environment import plan, install_all, discovered_servers, recipe, state_directory, lora_profile_from_identity
from .model_auth import AuthError, Credentials, WindowsVault, PROVIDERS, check as check_auth
from .lora_setup import conditions_text, require_user_key
from .h3_lora_profiles import H3_TERMS_URL


class SetupWindow:
    def __init__(self, root):
        self.root=root; self.events=queue.Queue(); self.cancelled=threading.Event()
        self.worker=None; self.controls=[]; self.readonly_controls=[]; self.credentials=Credentials(); self.vault=WindowsVault()
        error=None
        try:current=load()
        except (ValueError,OSError) as exc:current=RUNTIME;error=str(exc)
        servers=discovered_servers()
        if not current.configured and servers:
            existing=Path(servers[0]['root'])
            current=RuntimeConnection(existing,servers[0]['python'],servers[0]['endpoint'],True)
        root.title('Mosaic Refiner — 初回セットアップ');root.geometry('980x820')
        body=ttk.Frame(root,padding=16);body.pack(fill='both',expand=True)
        ttk.Label(body,text='推論環境を準備',font=('Yu Gothic UI',16)).pack(anchor='w')
        ttk.Label(body,text='お使いのComfyUIはありますか？ 既存環境を選ぶか、このアプリ用に新規導入します。',wraplength=940).pack(anchor='w',pady=6)
        journal=state_directory(current)/'state.json'
        managed_new=False
        record={}
        if journal.is_file():
            try:
                record=json.loads(journal.read_text(encoding='utf-8'))
                if not isinstance(record,dict):raise ValueError('導入記録の形式が不正です')
                managed_new=(record.get('root')==str(current.root) and record.get('managed_new') is True)
            except (OSError,ValueError):record={}
        self.mode=tk.StringVar(value='new' if managed_new or not current.configured else 'existing')
        modes=ttk.Frame(body);modes.pack(fill='x')
        for value,label in (('existing','既存ComfyUIを使う'),('new','ComfyUIを新規導入')):
            b=ttk.Radiobutton(modes,text=label,variable=self.mode,value=value,command=self.mode_changed)
            b.pack(side='left',padx=4);self.controls.append(b)
        self.root_dir=tk.StringVar(value=str(current.root));self.python=tk.StringVar(value=str(current.python))
        self.endpoint=tk.StringVar(value=current.endpoint)
        for label,var,chooser in (('ComfyUIの場所',self.root_dir,self.select_root),
               ('ComfyUIのPython',self.python,self.select_python),('APIアドレス',self.endpoint,None)):
            row=ttk.Frame(body);row.pack(fill='x',pady=3)
            ttk.Label(row,text=label,width=23).pack(side='left')
            entry=ttk.Entry(row,textvariable=var);entry.pack(side='left',fill='x',expand=True);self.controls.append(entry)
            if chooser:
                b=ttk.Button(row,text='選択',command=chooser);b.pack(side='left',padx=5);self.controls.append(b)
        ttk.Label(body,text='新規導入の推奨: NVIDIA Ampere以降・VRAM 24GB・RAM 64GB・空き120GB以上。最低条件はVRAM 12GB / RAM 32GBです。12〜23GBは未検証。NVIDIA R580以降のドライバーは事前に用意してください。',wraplength=940).pack(anchor='w',pady=8)
        row=ttk.Frame(body);row.pack(fill='x',pady=4)
        ttk.Label(row,text='今回導入する主LoRA（必須）',width=30).pack(side='left')
        previous=(record.get('main_lora') or record.get('completed_main_lora')) if record.get('root')==str(current.root) else None
        self.lora_choice=tk.StringVar(value='DecensoreH3' if lora_profile_from_identity(previous)=='restoration' else 'Unlocked V2')
        choice=ttk.Combobox(row,textvariable=self.lora_choice,values=('Unlocked V2','DecensoreH3'),state='readonly')
        choice.pack(side='left',fill='x',expand=True);self.controls.append(choice);self.readonly_controls.append(choice)
        choice.bind('<<ComboboxSelected>>',lambda event:self.refresh_models())
        tabs=ttk.Notebook(body);tabs.pack(fill='both',expand=True)
        requirements=ttk.Frame(tabs,padding=6);models=ttk.Frame(tabs,padding=6)
        tabs.add(requirements,text='PC要件と導入計画');tabs.add(models,text='モデルと利用条件')
        auth=ttk.Frame(tabs,padding=12);tabs.add(auth,text='本人のAPI認証')
        ttk.Label(auth,text='必須LoRAの取得には本人のCivitai APIキーが必要です。Hugging Faceは必要な場合だけ入力します。AIとのチャットへ貼り付けないでください。',
                  wraplength=900).pack(anchor='w',pady=6)
        self.auth_inputs={}
        for provider,label in (('huggingface','Hugging Face 読取り用token'),('civitai','Civitai APIキー')):
            row=ttk.Frame(auth);row.pack(fill='x',pady=8)
            ttk.Label(row,text=label,width=30).pack(side='left')
            value=tk.StringVar();self.auth_inputs[provider]=value
            entry=ttk.Entry(row,textvariable=value,show='*');entry.pack(side='left',fill='x',expand=True);self.controls.append(entry)
            for text,action in (('接続確認','check'),('OSに保存','save'),('保存を削除','delete')):
                button=ttk.Button(row,text=text,command=lambda p=provider,a=action:self.start_auth(p,a))
                button.pack(side='left',padx=3);self.controls.append(button)
            ttk.Button(auth,text=label+'の公式設定ページ',command=lambda p=provider:webbrowser.open(PROVIDERS[p][2])).pack(anchor='w')
        ttk.Label(auth,text='保存しないキーは、この画面を閉じるまでだけ使います。「OSに保存」はWindows資格情報マネージャーに保存します。\n'
                  '保存済みのキーは、次回の明示した取得操作でも使います。接続成功と各モデルの同意・作者承認は別です。',
                  wraplength=900).pack(anchor='w',pady=12)
        self.report=tk.Text(requirements,height=10,wrap='word',state='disabled',font=('Yu Gothic UI',10))
        self.report.pack(fill='both',expand=True)
        self.table=ttk.Treeview(models,columns=('size','license'),show='tree headings',height=8)
        self.table.heading('#0',text='モデル');self.table.heading('size',text='容量');self.table.heading('license',text='配布元の条件表示')
        self.table.column('#0',width=160);self.table.column('size',width=80);self.table.column('license',width=600)
        self.table.pack(fill='both',expand=True)
        for model,_ in resources(current,self.selected_lora()):
            self.table.insert('', 'end',iid=model.id,text=model.id,values=(f'{model.size_bytes/1e9:.2f} GB',model.license_label))
        ttk.Button(models,text='選択モデルの配布元・条件を開く',command=self.terms).pack(anchor='w',pady=4)
        ttk.Button(models,text='必須LoRAの作者・固定版・条件の確認状況',command=self.lora_terms).pack(anchor='w')
        ttk.Button(models,text='ComfyUI・ノード・Python・FFmpeg等の条件一覧',command=self.code_terms).pack(anchor='w')
        ttk.Label(body,text='出力はAIの推測です。Unlocked V2 / DecensoreH3の主LoRA一つと、本人のCivitai API認証が必須です。配布元の条件を確認して導入してください。導入計画に取得元と版を表示します。',wraplength=940).pack(anchor='w',pady=7)
        self.accepted=tk.BooleanVar(value=False);self.warning_ack=tk.BooleanVar(value=False)
        for text,var in (('各コード・モデル・FFmpeg等の配布元の利用条件を確認しました',self.accepted),
                         ('表示された未検証のGPU / RAM条件を確認して続行します',self.warning_ack)):
            b=ttk.Checkbutton(body,text=text,variable=var);b.pack(anchor='w');self.controls.append(b)
        actions=ttk.Frame(body);actions.pack(fill='x',pady=8)
        for label,kind in (('PC要件・導入計画を確認','plan'),('すべて導入 / 再開','install'),('導入済み環境を確認','check')):
            b=ttk.Button(actions,text=label,command=lambda k=kind:self.start(k));b.pack(side='left',padx=3);self.controls.append(b)
        self.cancel=ttk.Button(actions,text='中断',command=self.cancelled.set,state='disabled');self.cancel.pack(side='right')
        self.bar=ttk.Progressbar(body,maximum=100);self.bar.pack(fill='x',pady=5)
        self.status=tk.StringVar(value=error or '起動だけでは取得しません。PC要件・導入計画を確認してから開始してください。')
        ttk.Label(body,textvariable=self.status,wraplength=940).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(100,self.poll)

    def connection(self):
        if not self.root_dir.get().strip() or not self.python.get().strip():
            raise ValueError('ComfyUIの場所とPythonを指定してください')
        root=Path(self.root_dir.get().strip()).resolve()
        python=root/'.venv/Scripts/python.exe' if self.mode.get()=='new' else Path(self.python.get().strip())
        return RuntimeConnection(root,python,self.endpoint.get().strip(),True)

    def mode_changed(self):
        if self.mode.get()=='new':self.python.set(str(Path(self.root_dir.get())/'.venv/Scripts/python.exe'))
        self.status.set('新規導入先には空のフォルダーを選んでください。既存ComfyUIが動いている場合はその環境を選択します。')

    def select_root(self):
        value=filedialog.askdirectory(parent=self.root,title='ComfyUIの場所（新規は空の場所）',mustexist=False)
        if value:
            root=Path(value);self.root_dir.set(value)
            candidates=[root/'.venv/Scripts/python.exe',root.parent/'python_embeded/python.exe']
            self.python.set(str(next((p for p in candidates if p.is_file()),candidates[0])))

    def select_python(self):
        if self.mode.get()=='new':self.mode_changed();return
        value=filedialog.askopenfilename(parent=self.root,title='このComfyUIのPython',filetypes=[('Python','python.exe')])
        if value:self.python.set(value)

    def terms(self):
        selected=self.table.selection()
        if selected:webbrowser.open(next(m for m,_ in resources(RUNTIME,self.selected_lora()) if m.id==selected[0]).terms_url)

    def selected_lora(self):
        choices={'Unlocked V2':'unlocked_v2','DecensoreH3':'restoration'}
        if self.lora_choice.get() not in choices:raise ValueError('主LoRAを選択してください')
        return choices[self.lora_choice.get()]

    def refresh_models(self):
        self.accepted.set(False)
        for item in self.table.get_children():self.table.delete(item)
        for model,_ in resources(RUNTIME,self.selected_lora()):
            self.table.insert('', 'end',iid=model.id,text=model.id,values=(f'{model.size_bytes/1e9:.2f} GB',model.license_label))

    def lora_terms(self):
        window=tk.Toplevel(self.root)
        window.title('必須LoRAの条件確認')
        window.geometry('860x440')
        body=tk.Text(window,wrap='word',font=('Yu Gothic UI',10),padx=12,pady=12)
        body.pack(fill='both',expand=True)
        body.insert('1.0',conditions_text(self.selected_lora()))
        body.configure(state='disabled')

    def code_terms(self):
        links=[s['terms_url'] for s in recipe()['sources']]
        links += [H3_TERMS_URL,
                   'https://github.com/studio-yebisu/mosaic-refiner/blob/main/docs/use-conditions.md',
                  recipe()['ffmpeg']['terms_url'],'https://www.python.org/about/legal/',
                  'https://pytorch.org/','https://github.com/triton-lang/triton-windows',
                  'https://github.com/Comfy-Org/comfy-kitchen','https://github.com/Comfy-Org/comfy-aimdo']
        self.show_report('配布元の条件一覧（READMEの導入案内にも記載）\n\n'+'\n'.join(links))

    def show_report(self,text):
        self.report.configure(state='normal');self.report.delete('1.0','end');self.report.insert('end',text);self.report.configure(state='disabled')

    def entered_credentials(self):
        credentials=Credentials.saved(self.vault)
        for provider,value in self.credentials.tokens.items():credentials.set(provider,value)
        for provider,var in self.auth_inputs.items():
            token=var.get().strip()
            if token:credentials.set(provider,token)
        return credentials

    def start_auth(self,provider,action):
        if self.worker and self.worker.is_alive():return
        try:
            credentials=Credentials() if action=='delete' else self.entered_credentials()
            if action!='delete' and provider not in credentials.tokens:
                self.status.set('この配布元の認証情報を本人が入力してください');return
        except (ValueError,OSError) as exc:self.status.set(str(exc));return
        for widget in self.controls:widget.configure(state='disabled')
        self.status.set('本人のAPI認証を確認中…')
        def work():
            try:
                if action=='check':check_auth(provider,credentials)
                elif action=='save':self.vault.save(provider,credentials.tokens[provider])
                else:self.vault.delete(provider)
                self.events.put(('auth',(provider,action,credentials)))
            except AuthError as exc:
                self.events.put(('error',str(exc)))
            except Exception:
                self.events.put(('error','API認証を確認・保存できません。キー、公式ページの権限、OS資格情報を確認してください。'))
        self.worker=threading.Thread(target=work,daemon=True);self.worker.start()

    def start(self,kind):
        if self.worker and self.worker.is_alive():return
        try:
            connection=self.connection()
            lora_profile=self.selected_lora()
            credentials=self.entered_credentials() if kind=='install' else None
            if kind=='install':require_user_key(credentials)
        except (ValueError,OSError) as exc:self.status.set(str(exc));return
        if kind=='install' and not self.accepted.get():
            self.status.set('配布元の利用条件を確認してチェックを入れてください');return
        new=self.mode.get()=='new';accepted=self.accepted.get();ack=self.warning_ack.get()
        self.cancelled.clear()
        for w in self.controls:w.configure(state='disabled')
        self.cancel.configure(state='normal');self.status.set('PC要件と所有環境を確認中…');self.bar['value']=0
        def work():
            try:
                progress=lambda **kw:self.events.put(('progress',kw))
                if kind=='plan':result=plan(connection,new,self.cancelled.is_set,progress,lora_profile=lora_profile)
                elif kind=='install':result=install_all(connection,new,accepted,ack,None,self.cancelled.is_set,progress,
                                                       credentials=credentials,lora_profile=lora_profile)
                else:result={'nodes':inspect_nodes(connection),'models':resource_status(connection,cancelled=self.cancelled.is_set,progress=progress,lora_profile=lora_profile)}
                self.events.put(('done',(kind,result)))
            except KeyboardInterrupt:self.events.put(('error','中断しました。途中データを保持します。再開は「すべて導入 / 再開」から行ってください。'))
            except Exception as exc:self.events.put(('error',str(exc)))
        self.worker=threading.Thread(target=work,daemon=True);self.worker.start()

    def poll(self):
        while True:
            try:kind,data=self.events.get_nowait()
            except queue.Empty:break
            if kind=='progress':
                self.status.set(data['phase']+' / '+data.get('model',''))
                if data.get('total'):self.bar['value']=data.get('downloaded',0)/data['total']*100
            else:
                for w in self.controls:w.configure(state='normal')
                for w in self.readonly_controls:w.configure(state='readonly')
                self.cancel.configure(state='disabled')
                if kind=='error':self.status.set(data);continue
                if kind=='auth':
                    provider,action,credentials=data
                    if action=='delete':
                        self.credentials.tokens.pop(provider,None);self.auth_inputs[provider].set('')
                        self.status.set('保存した認証情報を削除しました')
                    else:
                        self.credentials.set(provider,credentials.tokens[provider])
                        self.auth_inputs[provider].set('')
                        self.status.set('接続を確認しました。各モデルの取得権限は別途確認します。' if action=='check'
                                        else 'Windows資格情報マネージャーへ保存しました')
                    continue
                action,result=data
                if action=='plan':
                    hw=result['hardware'];gpu=hw['hardware']['gpus']
                    lines=[f"導入方式: {result['mode']} / {result['root']}",f"GPU: {gpu}",
                           f"RAM: {hw['hardware']['ram_gib']:.1f}GiB",hw['tested_configuration'],*hw['blockers'],*hw['warnings'],
                           '取得先の空き容量: '+str(hw['storage']),
                           '必須LoRA: '+str(result['main_lora'])+' / 本人のCivitai API認証が必要です。',
                           'LoRAの固定版: '+json.dumps(result['lora_model'],ensure_ascii=False),
                           '処理動画・既存出力は使いません。導入後も実動画の完走・画質は別の確認です。',
                           '今回の導入版（再開時は固定）: '+json.dumps(result.get('resolution',{}),ensure_ascii=False,indent=2)]
                    self.show_report('\n\n'.join(lines));self.status.set('PC要件を満たしています。配布元の利用条件を確認して導入を開始してください。' if hw['can_install'] else '要件が不足しています。モデル取得を開始できません。')
                elif action=='install':
                    self.status.set('依存・モデル・GPUカーネル・ノード登録を確認しました。Web/Windowsアプリを起動し直してください。' if result['state']=='dependencies_ready' else result.get('next',result['state']))
                    self.show_report(json.dumps(result,ensure_ascii=False,indent=2))
                else:
                    ready=result['nodes']['nodes_verified'] and all(r['state']=='verified' for r in result['models'])
                    self.status.set('ノードとモデルを確認しました。動画を選んで補正を開始できます。' if ready else '不足またはhash未確認のノード・モデルがあります。表示内容を確認してください。')
                    self.show_report(json.dumps(result,ensure_ascii=False,indent=2))
        self.root.after(100,self.poll)

    def close(self):
        if self.worker and self.worker.is_alive():
            self.cancelled.set();self.status.set('現在の操作を中断してから閉じます');self.root.after(200,self.close)
        else:self.root.destroy()


def main():
    root=tk.Tk();SetupWindow(root);root.mainloop()


if __name__=='__main__':main()
