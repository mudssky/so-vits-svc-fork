import io
import os
import argparse
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from inference.infer_tool import Svc
import logging
import json
import matplotlib.pyplot as plt
import parselmouth

parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, help='set gradio user', default=None)
parser.add_argument("--password", type=str, help='set gradio password', default=None)
cmd_opts = parser.parse_args()

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def load_model_func(ckpt_name,cluster_name,config_name):
    global model, cluster_model_path
    
    config_path = "configs/" + config_name

    with open(config_path, 'r') as f:
        config = json.load(f)
    spk_dict = config["spk"]
    spk_name = config.get('spk', None)
    if spk_name:
        spk_choice = next(iter(spk_name))
    else:
        spk_choice = "未检测到音色"

    ckpt_path = "logs/44k/" + ckpt_name
    cluster_path = "logs/44k/" + cluster_name
    if cluster_name == "no_clu":
            model = Svc(ckpt_path,config_path)
    else:
            model = Svc(ckpt_path,config_path,cluster_model_path=cluster_path)

    spk_list = list(spk_dict.keys())
    return "模型加载成功", gr.Dropdown.update(choices=spk_list, value=spk_choice),

def load_options():
    file_list = os.listdir("logs/44k")
    ckpt_list = []
    cluster_list = []
    for ck in file_list:
        if os.path.splitext(ck)[-1] == ".pth" and ck[0] != "k":
            ckpt_list.append(ck)
        if ck[0] == "k":
            cluster_list.append(ck)
    if not cluster_list:
        cluster_list = ["你没有聚类模型"]
    return choice_ckpt.update(choices = ckpt_list), config_choice.update(choices = os.listdir("configs")), cluster_choice.update(choices = cluster_list)

def get_pitch(wav_data, mel, hparams):
    """
    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    time_step = hparams['hop_size'] / hparams['audio_sample_rate']
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    # if hparams['hop_size'] == 128:
    #     pad_size = 4
    # elif hparams['hop_size'] == 256:
    #     pad_size = 2
    # else:
    #     assert False

    f0 = parselmouth.Sound(wav_data, 44100).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    # lpad = pad_size * 2
    # rpad = len(mel) - len(f0) - lpad
    # f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    # # mel and f0 are extracted by 2 different libraries. we should force them to have the same length.
    # # Attention: we find that new version of some libraries could cause ``rpad'' to be a negetive value...
    # # Just to be sure, we recommend users to set up the same environments as them in requirements_auto.txt (by Anaconda)
    # delta_l = len(mel) - len(f0)
    # assert np.abs(delta_l) <= 8
    # if delta_l > 0:
    #     f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    # f0 = f0[:len(mel)]
    pad_size=(int(len(wav_data) // hparams['hop_size']) - len(f0) + 1) // 2
    f0 = np.pad(f0,[[pad_size,len(mel) - len(f0) - pad_size]], mode='constant')
    pitch_coarse = f0_to_coarse(f0, hparams)
    return f0, pitch_coarse

def f0_plot(ckpt_name,cluster_name,config_name):
    config_path = "configs/" + config_name
    ckpt_path = "logs/44k/" + ckpt_name
    cluster_path = "logs/44k/" + cluster_name
    if cluster_name == "no_clu":
        svc_model = Svc(ckpt_path, config_path)
    else:
        svc_model = Svc(ckpt_path, config_path, cluster_model_path=cluster_path)
    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp.wav")
    wav_data, mel = svc_model.wav2spec(wav_path)
    f0_gen, _ = get_pitch(wav_data, mel, hparams)
    f0_tst[f0_tst==0]=np.nan#ground truth f0
    f0_pred[f0_pred==0]=np.nan#f0 pe predicted
    f0_gen[f0_gen==0]=np.nan#f0 generated
    fig=plt.figure(figsize=[15,5])
    plt.plot(np.arange(0,len(f0_tst)),f0_tst,color='black')
    plt.plot(np.arange(0,len(f0_pred)),f0_pred,color='orange')
    plt.plot(np.arange(0,len(f0_gen)),f0_gen,color='red')
    plt.axhline(librosa.note_to_hz('C4'),ls=":",c="blue")
    plt.axhline(librosa.note_to_hz('G4'),ls=":",c="green")
    plt.axhline(librosa.note_to_hz('C5'),ls=":",c="orange")
    plt.axhline(librosa.note_to_hz('F#5'),ls=":",c="red")
    #plt.axhline(librosa.note_to_hz('A#5'),ls=":",c="black")
    return fig

def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    with sf.SoundFile("temp.wav") as wav_file:
        bit_depth = wav_file.subtype
        if bit_depth != "PCM_16":
            return "上传的音频不是16位wav，请重新上传", None
        print(bit_depth)
    # print(audio.shape,sampling_rate)
    # duration = audio.shape[0] / sampling_rate
    #if duration > 90:
    #    return "请上传小于90s的音频，需要转换长音频请本地进行转换", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = "temp.wav"
    sf.write(out_wav_path, audio, 16000, format="wav")
    print( cluster_ratio, auto_f0, noise_scale)
    _audio = model.slice_inference(out_wav_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale)
    return "Success", (44100, _audio)

# read ckpt list
file_list = os.listdir("logs/44k")
ckpt_list = []
cluster_list = []
for ck in file_list:
    if os.path.splitext(ck)[-1] == ".pth" and ck[0] != "k" and ck[:2] != "D_":
        ckpt_list.append(ck)
    if ck[0] == "k":
        cluster_list.append(ck)
if not cluster_list:
    cluster_list = ["你没有聚类模型"]
    #print("no clu")

app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""
                sovits4.0 webui
                """)
            choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value="no_model")
            config_choice = gr.Dropdown(label="配置文件", choices=os.listdir("configs"), value="no_config")
            cluster_choice = gr.Dropdown(label="选择聚类模型", choices=cluster_list, value="no_clu")
            refresh = gr.Button("刷新选项")
            loadckpt = gr.Button("加载模型", variant="primary")
            
            sid = gr.Dropdown(label="音色", value="speaker0")
            model_message = gr.Textbox(label="Output Message")
            
            refresh.click(load_options,[],[choice_ckpt, config_choice,cluster_choice])
            loadckpt.click(load_model_func,[choice_ckpt,cluster_choice,config_choice],[model_message, sid])
            
            gr.Markdown(value="""
                请稍等片刻，模型加载大约需要10秒。后续操作不需要重新加载模型
                """)
            
            
            
            vc_input3 = gr.Audio(label="上传音频")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例，0-1之间，默认为0不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
            auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
            
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
            #f0_output = gr.Image(label = "f0")
            #f0_generate = gr.Button("生成f0图像（请在推理成功后生成）", variant="primary")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale], [vc_output1, vc_output2])
        #f0_generate.click(f0_plot,[choice_ckpt,cluster_choice,config_choice],[f0_output])

        app.launch()




