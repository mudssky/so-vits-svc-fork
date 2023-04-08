import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging

import subprocess
from scipy.io import wavfile
import torch
import time
import pathlib
import json
import re

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

# 模型文件
model = None
# 可选择的说话人列表
# spk_list = []
# 显卡列表
cuda_list = []
# 模型文件夹位置
mods_path = 'mods'


def get_cuda_list():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cuda_list.append("cuda:{}".format(i))


# 输入音频转换
def vc_fn(sid, input_audio, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale,
          pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling, output_path,
          input_filename=''):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if not output_path:
            return "You need to set output path", None
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db,
                                       cluster_ratio, auto_f0, noise_scale, pad_seconds,
                                       cl_num, lg_num, lgr_num, F0_mean_pooling)
        model.clear_empty()
        os.remove(temp_path)
        # 构建保存文件的路径，并保存到results文件夹内
        try:
            if input_filename:
                input_filename = input_filename+'_'
            timestamp = str(int(time.time()))
            output_file = os.path.join(
                output_path, sid + "_" + input_filename + timestamp + ".wav")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(output_file, 'wb') as f:
                soundfile.write(f, _audio,
                                model.target_sample, format="wav")
            return "Success", (model.target_sample, _audio)
        except Exception as e:
            print(e)
            return "自动保存失败，请手动保存，音乐输出见下", (model.target_sample, _audio)
    except Exception as e:
        print(e)
        print(e.with_traceback())
        return "异常信息:"+str(e)+"\n请排障后重试", None

# 调用tts


def tts_func(_text, _rate):
    # 使用edge-tts把文字转成音频
    # voice = "zh-CN-XiaoyiNeural"#女性，较高音
    # voice = "zh-CN-YunxiNeural"#男性
    voice = "zh-CN-YunxiNeural"  # 男性
    output_file = _text[0:10]+".wav"
    # communicate = edge_tts.Communicate(_text, voice)
    # await communicate.save(output_file)
    if _rate >= 0:
        ratestr = "+{:.0%}".format(_rate)
    elif _rate < 0:
        ratestr = "{:.0%}".format(_rate)  # 减号自带

    p = subprocess.Popen(["edge-tts",
                          "--text", _text,
                         "--write-media", output_file,
                          "--voice", voice,
                          "--rate="+ratestr], shell=True,
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    p.wait()
    return output_file

# 文字转音频


def vc_fn2(sid, vc_transform, auto_f0, cluster_ratio,
           slice_db, noise_scale,
           pad_seconds, cl_num, lg_num, lgr_num,
           text2tts, tts_rate, F0_mean_pooling, output_path):
    # 使用edge-tts把文字转成音频
    output_file = tts_func(text2tts, tts_rate)

    # 调整采样率
    sr2 = 44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2 = text2tts[0:10]+"_44k"+".wav"
    wavfile.write(save_path2, sr2,
                  (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                  )

    # 读取音频
    sample_rate, data = gr_pu.audio_from_file(save_path2)
    vc_input = (sample_rate, data)

    a, b = vc_fn(sid, vc_input, vc_transform, auto_f0, cluster_ratio, slice_db,
                 noise_scale, pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling,
                 output_path)
    os.remove(output_file)
    os.remove(save_path2)
    return a, b


def batch_vc(sid, input_audio_folder, vc_transform, auto_f0, cluster_ratio, slice_db,
             noise_scale,
             pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling, output_path,
             progress=gr.Progress()):
    audio_list = get_audio_list(input_audio_folder)
    for audio_path in progress.tqdm(audio_list, desc="convert audio"):
        audio_data = gr_pu.audio_from_file(audio_path)
        filename = pathlib.PurePath(audio_path).name
        name = filename.split('.')[0]
        print(audio_path)
        # print(audio_data)
        vc_fn(sid, audio_data, vc_transform, auto_f0, cluster_ratio, slice_db,
              noise_scale, pad_seconds, cl_num, lg_num, lgr_num, F0_mean_pooling,
              output_path, input_filename=name)
    return '全部转换完成'


def get_config_file(model_foldername):
    model_folderpath = os.path.join(mods_path, model_foldername)
    mod_folder = pathlib.Path(model_folderpath)
    config_list = conver_path_glob(mod_folder.glob("*.json"))
    return config_list


def get_kmeans_cluster_model(model_foldername):

    model_folderpath = os.path.join(mods_path, model_foldername)
    mod_folder = pathlib.Path(model_folderpath)
    kmeans_cluster_model_list = conver_path_glob(mod_folder.glob('*.pt'))
    return kmeans_cluster_model_list


def get_model_folder_list():
    global model_folder_list
    mflist = []
    mod_folder_list = os.listdir(mods_path)
    for folder in mod_folder_list:
        mod_folder_dir = pathlib.Path(os.path.join(mods_path, folder))
        if mod_folder_dir.glob("*.json") and mod_folder_dir.glob("*.pth"):
            mflist.append(folder)
    model_folder_list = mflist
    return mflist


def get_model_list(foldername):
    model_folderpath = os.path.join(mods_path, foldername)
    model_list = conver_path_glob(pathlib.Path(
        model_folderpath).glob('*.pth'))
    return model_list


def conver_path_glob(pathglob):
    if pathglob:
        return [os.fspath(p) for p in pathglob]
    return pathglob


def load_model_func(model_path, config_path, kmeans_cluster_model_path):
    global model
    with open(config_path, 'r') as f:
        config = json.load(f)
    spk_dict = config["spk"]
    spk_name = config.get('spk', None)
    if spk_name:
        spk = next(iter(spk_name))
    else:
        spk = "未检测到音色"
    if not kmeans_cluster_model_path:
        model = Svc(model_path, config_path)
    else:
        model = Svc(model_path, config_path,
                    cluster_model_path=kmeans_cluster_model_path)
    spk_list = list(spk_dict.keys())
    return gr.Dropdown.update(value=spk, choices=spk_list), "模型加载成功"


def handle_model_path_select(choice_model_folder):
    model_list = get_model_list(choice_model_folder)
    default_model = get_first_element(model_list)
    kmeans_cluster_model_list = get_kmeans_cluster_model(choice_model_folder)
    default_kmeans = get_first_element(kmeans_cluster_model_list)
    config_list = get_config_file(choice_model_folder)
    default_config = get_first_element(config_list)
    return [gr.Dropdown.update(choices=model_list, value=default_model),
            gr.Dropdown.update(choices=config_list, value=default_config),
            gr.Dropdown.update(
                choices=kmeans_cluster_model_list, value=default_kmeans),
            ]


def before_app_create():
    # load_options()
    pass


def is_audio(filename):
    patten = r'[\s\S]*\.(mp3|wav|flac|m4a)'
    if re.match(patten, filename):
        return True
    return False


def get_first_element(list):
    if list:
        return list[0]
    return 'no_element'


def recursive_scandir(path, callback):
    for entry in os.scandir(path):
        if entry.is_file():
            if callback:
                callback(entry.path)
        elif entry.is_dir():
            recursive_scandir(entry.path, callback=None)


def get_audio_list(path):
    audio_list = []

    def collect_audio(pathname):
        if is_audio(pathname):
            audio_list.append(pathname)
    recursive_scandir(path, collect_audio)
    return audio_list


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Sovits4.0"):
            gr.Markdown(value="""
                Sovits4.0 WebUI
                模型，聚类文件，config放到mods目录下创建对应的文件夹，就会自动读取
                mods/model_name/*.pth
                mods/model_name/config.json
                mods/model_name/*.pt
                """)

            choice_model_folder = gr.Dropdown(
                label="模型目录选择", choices=get_model_folder_list(),
                info="选择目录后，会自动读取目录下的模型文件，配置文件和聚类模型",
                interactive=True)

            choice_model = gr.Dropdown(
                label="模型文件选择", interactive=True)
            choice_config = gr.Dropdown(
                label="配置文件选择",
                interactive=True)
            # choice_config.change(handle_config_change,
            #                      inputs=[choice_config])
            choice_kmeans_model = gr.Dropdown(
                label="聚类模型选择")
            choice_model_folder.change(handle_model_path_select,
                                       inputs=[choice_model_folder],
                                       outputs=[choice_model, choice_config,
                                                choice_kmeans_model])

            device = gr.Dropdown(label="推理设备，默认为自动选择cpu和gpu", choices=[
                                 "Auto", *cuda_list, "cpu"], value="Auto",
                                 interactive=True)
            gr.Markdown(value="""
                <font size=3>选择完模型后点击加载模型：</font>
                """)
            load_model_button = gr.Button(value="加载模型", variant="primary")
            refresh = gr.Button("刷新选项")

            def load_options():
                print('load option')
                new_model_folder_list = get_model_folder_list()
                default_model_folder = get_first_element(new_model_folder_list)
                return choice_model_folder.update(
                    choices=new_model_folder_list, value=default_model_folder)

            refresh.click(load_options, outputs=[choice_model_folder])
            sid = gr.Dropdown(
                label="音色（说话人）")
            sid_output = gr.Textbox(label="Output Message")

            load_model_button.click(
                load_model_func,
                inputs=[choice_model, choice_config, choice_kmeans_model],
                outputs=[sid, sid_output])

            text2tts = gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
            tts_rate = gr.Number(label="tts语速", value=0)

            vc_input3 = gr.Audio(label="上传音频")
            bacth_convert_folder = gr.Textbox(
                label="批量转换音频的目录，会自动找到目录中的音频文件进行转换", value="input_folder")
            vc_transform = gr.Number(
                label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
            cluster_ratio = gr.Number(
                label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）",
                value=0)
            auto_f0 = gr.Checkbox(
                label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）",
                value=False)
            F0_mean_pooling = gr.Checkbox(
                label="是否对F0使用均值滤波器(池化)，对部分哑音有改善。注意，启动该选项会导致推理速度下降，默认关闭", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(
                label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
            cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒/s", value=0)
            pad_seconds = gr.Number(
                label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
            lg_num = gr.Number(
                label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s",
                value=0)
            lgr_num = gr.Number(
                label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75,
                interactive=True)
            vc_submit = gr.Button("音频直接转换", variant="primary")
            vc_submit2 = gr.Button("文字转音频+转换", variant="primary")
            bacth_convert_button = gr.Button("批量转换音频", variant="primary")
            output_path = gr.Textbox(label="输出路径", value="results")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform, auto_f0, cluster_ratio,
                                slice_db, noise_scale, pad_seconds, cl_num, lg_num,
                                lgr_num, F0_mean_pooling,
                                output_path], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [sid, vc_transform, auto_f0, cluster_ratio,
                                  slice_db, noise_scale, pad_seconds, cl_num, lg_num,
                                  lgr_num, text2tts, tts_rate, F0_mean_pooling,
                                  output_path],
                         [vc_output1, vc_output2])
        bacth_convert_button.click(batch_vc, [sid, bacth_convert_folder, vc_transform,
                                              auto_f0,
                                              cluster_ratio, slice_db, noise_scale,
                                              pad_seconds, cl_num, lg_num, lgr_num,
                                              F0_mean_pooling,
                                              output_path], [vc_output1])

app.queue(concurrency_count=20).launch(server_name="0.0.0.0", server_port=7863)
