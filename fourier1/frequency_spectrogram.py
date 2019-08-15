import numpy as np
import os


def run():
        base = "/net/projects/scratch/summer/valid_until_31_January_2020/ann4auto/Combined_Chunks/Acc_Vel/0.01s_duration/0.0s_overlap"
        experimental_conditions = ["/TS_IO_Neu_Neu_combined/","/TS_Neu_Neu_NIO_combined/","/TS_NIO_Neu_Neu_combined/","/TS_NIO_NIO_NIO_combined/",
                                   "/TS_Neu_Neu_Neu_combined/","/TS_Neu_NIO_Neu_combined/","/TS_NIO_NIO_Neu_combined/"]

        for condition in experimental_conditions:
            filepaths = []#get filepaths
            for x in os.listdir(base+condition):
                filepaths.append(base+condition+x)

            f_max = 100#Sample length //2
            n_samples = len(filepaths)
            n_channels = 9#Channel 10 is speed and not of interest

            power = np.zeros((n_channels,n_samples,f_max))#Init output tensor
            for i, name in enumerate(filepaths):
                chunk = np.load(name)
                print(str((i/len(filepaths))*100)+"%")
                for x in range(n_channels):
                    snippet = chunk[:,x]
                    val = np.abs(np.fft.fft(snippet).real[:f_max])
                    power[x][i] = val

            for x in range(n_channels):
                try:
                    os.mkdir(os.getcwd()+condition)                    
                except Exception as e:
                    print("Error creating directory")
                    print(e)
                try:
                    np.save(os.getcwd()+condition+"spectrogram_channel_"+str(x)+".npy", power[x])
                except Exception as e:
                    print(e)
	


if __name__ == "__main__":
    run()
