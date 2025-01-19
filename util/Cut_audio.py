import os  # 文件系统操作对象
from pydub import AudioSegment
import glob

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        print("success")

def audio_sound(path, length):
    files = [path + "/" + x for x in os.listdir(path) if os.path.isdir(path + "/" + x)]
    for folder in files:
        print(os.path.basename(folder))
        # save_file_path = os.path.join(r"D://seven77//qpyCode//Mel-CAM//audio_cut",os.path.basename(folder))
        # print(save_file_path)
        # if not os.path.exists(save_file_path):
        #     os.makedirs(save_file_path)
        for audio in glob.glob(folder + '/*.wav'):
            # 读取文件有很多方式，有直接from_file(),也有from_mp3()、from_wav(),下面的两个读取语句是等价的：
            # sound = AudioSegment.from_file("mp3/正常.m4a", "m4a")
            # sound = AudioSegment.from_mp3("mp3/15test.mp3")
            sound = AudioSegment.from_file(audio)
            print(len(sound))
            print('时长：{} s'.format(len(sound) / 1000))
            chunk_num = int(len(sound) / 1000 / length)
            start_time = 0
            end_time = length * 1000
            for n in range(chunk_num):
                # 切割文件
                part = sound[start_time:end_time]
                # 保存路径  D://seven77//qpyCode//Mel-CAM//audio_cut/0\1-11687-A-47_1.wav
                FileName = os.path.basename(audio)[0:12] + "_" + str(n + 1) + ".wav"
                # 保存路径  D://seven77//qpyCode//Mel-CAM//audio_cut/0\1-11687-A-47_1.wav
                out_path = os.path.join("../Cut_deepShip", os.path.basename(folder), FileName)
                # print(os.path.dirname(out_path).rsplit('/', 1)[0])
                create_folder(os.path.dirname(out_path))
                print(out_path)
                # 保存文件
                part.export(out_path, format="wav")
                start_time += length
                end_time += length


if __name__ == '__main__':
    # 音频所在目录
    dir_path = "../../deepship" # ../../Mel-CAM/audio_pre
    # 每段剪辑长度 1s
    audio_length = 1
    # 执行
    audio_sound(dir_path, audio_length)
