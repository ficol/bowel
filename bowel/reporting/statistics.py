import argparse
import datetime

import pandas as pd
import git
import matplotlib.pyplot as plt
import numpy as np

from bowel.utils.audio_utils import get_wav_length

class Report:
    """A class to calculate statistics and create report from inference results.
    """

    def __init__(self, csv_file, wav_file=None):
        """Report constructor.

        Args:
            csv_file (str): path to csv file
            wav_file (str): path to wav file
        """
        self.report = {}
        try:
            git_hash = git.Repo(
                search_parent_directories=True).head.object.hexsha
        except:
            git_hash = '0'
        f = open(csv_file, 'r')
        lines = f.readlines()
        self.report['Basic informations'] = {'Date': datetime.date.today().strftime("%d %B %Y"),
                                            'Patient ID': '',
                                            'Age, years': '',
                                            'Height, cm': '',
                                            'Mass, kg': '',
                                            'Symptoms': '',
                                            'Diagnoses': '',
                                            'Medication': '',
                                            'Git hash': git_hash,
                                            'Inference hash': lines[0].split('=')[1][:-1] if len(lines[0].split('=')) >= 2 else '',
                                            'Model type': lines[1].split('=')[1][:-1] if len(lines[1].split('=')) >= 2 else '',
                                            'Model version': lines[2].split('=')[1][:-1] if len(lines[2].split('=')) >= 2 else ''
                                            }
        f.close()
        self.sounds = pd.read_csv(csv_file, comment='#').to_dict('records')
        self.duration = get_wav_length(wav_file) if wav_file is not None else float(lines[3].split('=')[1])
        self.freqs_per_minute = self.__get_freqs_per_minute(self.sounds)
        self.durations_per_minute = self.__get_durations_per_minute(self.sounds)
        self.durations = self.__get_durations(self.sounds)
        self.intervals = self.__get_intervals(self.sounds)
        if not self.durations:
            self.durations = [0]
        self.report['Main results'] = self.get_statistics_report()
        self.report['Plots'] = self.get_plots()

    def save(self, xlsx_file):
        """Save report to xlsx.
        """
        df = pd.DataFrame(
            data=self.report['Basic informations'] | self.report['Main results'], index=[0]).T
        with pd.ExcelWriter(xlsx_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, float_format='%.2f',
                        header=False, sheet_name='report')
            workbook = writer.book
            format = workbook.add_format()
            format.set_align('left')

            writer.sheets['report'].set_column('A:A', 65, format)
            writer.sheets['report'].set_column('B:B', 45, format)
            path_to_images = '/tmp/'

            title = 'Bowel sounds per minute'
            plt.plot(*zip(*self.report['Plots'][title]))
            plt.title(title)
            plt.xlabel('Time[min]')
            plt.ylabel('Amount')
            plt.savefig(path_to_images + title)
            plt.clf()
            worksheet = workbook.add_worksheet('Sounds per minute')
            worksheet.insert_image('E2', path_to_images + title + '.png')
            worksheet.write(0, 0, 'Time[min]')
            worksheet.write(0, 1, 'Average amount per minute')
            for i, (x,y) in enumerate(self.report['Plots'][title]):
                worksheet.write(i + 1, 0, x)
                worksheet.write(i + 1, 1, y)
            writer.save

            title = 'Bowel sound durations per minute'
            plt.plot(*zip(*self.report['Plots'][title]))
            plt.title(title)
            plt.xlabel('Time[min]')
            plt.ylabel('Duration[s]')
            plt.savefig(path_to_images + title)
            plt.clf()
            worksheet = workbook.add_worksheet('Durations per minute')
            worksheet.insert_image('E2', path_to_images + title + '.png')
            worksheet.write(0, 0, 'Time[min]')
            worksheet.write(0, 1, 'Average duration per minute [s]')
            for i, (x,y) in enumerate(self.report['Plots'][title]):
                worksheet.write(i + 1, 0, x)
                worksheet.write(i + 1, 1, y)
            writer.save

            title = 'Histogram of bowel sounds per minute'
            plt.hist(3 * self.report['Plots'][title], bins=20)
            plt.title(title)
            plt.xlabel('Bowel sounds per minute')
            plt.ylabel('Amount of 3 minutes intervals')
            plt.savefig(path_to_images + title)
            plt.clf()
            worksheet = workbook.add_worksheet('Hist sounds per minute')
            worksheet.insert_image('E2', path_to_images + title + '.png')
            worksheet.write(0, 0, 'Time[min]')
            worksheet.write(0, 1, 'Average amount per minute')
            for i, x in enumerate(self.report['Plots'][title]):
                worksheet.write(i + 1, 0, 3 * (i + 1))
                worksheet.write(i + 1, 1, x)
            writer.save

            title = 'Histogram of bowel sounds duration'
            plt.hist(self.report['Plots'][title], bins=20, range=[0,200])
            plt.title(title)
            plt.xlabel('Bowel sounds durations[ms]')
            plt.ylabel('Bowel sounds')
            plt.savefig(path_to_images + title)
            plt.clf()
            worksheet = workbook.add_worksheet('Hist durations per minute')
            worksheet.insert_image('E2', path_to_images + title + '.png')
            worksheet.write(0, 1, 'Duration[ms]')
            for i, x in enumerate(self.report['Plots'][title]):
                worksheet.write(i + 1, 1, x)
            writer.save

            title = 'Scatterplot of bowel sounds per minute vs duration'
            plt.scatter(*zip(*self.report['Plots'][title]))
            plt.title(title)
            plt.xlabel('Amount')
            plt.ylabel('Duration[s]')
            plt.savefig(path_to_images + title)
            plt.clf()
            worksheet = workbook.add_worksheet('Scatterplot sounds vs duration')
            worksheet.insert_image('E2', path_to_images + title + '.png')
            worksheet.write(0, 0, 'Time[min]')
            worksheet.write(0, 1, 'Sounds amount')
            worksheet.write(0, 2, 'Duration[s]')
            for i, (x,y) in enumerate(self.report['Plots'][title]):
                worksheet.write(i + 1, 0, 3 * (i + 1))
                worksheet.write(i + 1, 1, x)
                worksheet.write(i + 1, 2, y)
            writer.save

    def get_statistics_report(self):
        """get statistics from infer results.

        Returns:
            dict: Dict containing statistics calculated from infer results.
        """
        rmssd = np.sqrt(np.mean(np.array(self.intervals)**2))
        statistics = {'Recording length, minutes': self.duration / 60,
                      'Recording length, hours:minutes:seconds': str(datetime.timedelta(seconds=int(self.duration))),
                      'Bowel sounds identified, total count': len(self.sounds),
                      'Bowel_sounds_per minute, total': len(self.sounds) / (self.duration / 60),
                      'Frequency analysis in three-minute periods': 'Not Available',
                      'Bowel sounds per minute, mean': np.mean(self.freqs_per_minute),
                      'Bowel sounds per minute, standard deviation': np.std(self.freqs_per_minute),
                      'Bowel sounds per minute, median': np.median(self.freqs_per_minute),
                      'Bowel sounds per minute, 1st quartile': np.percentile(self.freqs_per_minute, 25),
                      'Bowel sounds per minute, 3rd quartile': np.percentile(self.freqs_per_minute, 75),
                      'Bowel sounds per minute, 1st decile': np.percentile(self.freqs_per_minute, 10),
                      'Bowel sounds per minute, 9th decile': np.percentile(self.freqs_per_minute, 90),
                      'Bowel sounds per minute, minimum': np.min(self.freqs_per_minute),
                      'Bowel sounds per minute, maximum': np.max(self.freqs_per_minute),
                      '% of bowel sounds followed by another bowel sound within 50 ms': len([interval for interval in self.intervals if interval <= 0.05 + 1e-5]) * 100 / max(1, len(self.sounds)),
                      '% of bowel sounds followed by another bowel sound within 100 ms': len([interval for interval in self.intervals if interval <= 0.1 + 1e-5]) * 100 / max(1, len(self.sounds)),
                      '% of bowel sounds followed by another bowel sound within 200 ms': len([interval for interval in self.intervals if interval <= 0.2 + 1e-5]) * 100 / max(1, len(self.sounds)),
                      'Duration analysis approximated to 10 ms': 'Not Available',
                      'Duration, mean': np.mean(self.durations),
                      'Duration standard deviation': np.std(self.durations),
                      'Duration, median': np.median(self.durations),
                      'Duration, 1st quartile': np.percentile(self.durations, 25),
                      'Duration, 3rd quartile': np.percentile(self.durations, 75),
                      'Duration, 1st decile': np.percentile(self.durations, 10),
                      'Duration, 9rd decile': np.percentile(self.durations, 90),
                      'Duration, min': np.min(self.durations),
                      'Duration, max': np.max(self.durations),
                      'Power analysis': 'Not Available',
                      'Root mean square of the successive differences (RMSSD)': rmssd,
                      'Logarithm of RMSSD': np.log(rmssd),
                      'Standard deviation of bowel sound intervals (SDNN)': np.std(self.intervals),
                      'Porta\'s index': self.__get_portas_index(self.intervals),
                      'Guzik\'s index': self.__get_guziks_index(self.intervals)
                      }
        return statistics

    def get_plots(self):
        """get list of points for plots.

        Returns:
            dict[list]: Dict containing lists of points of plots.
        """
        plots = {'Bowel sounds per minute': self.__get_time_plot(self.freqs_per_minute),
                 'Bowel sound durations per minute': self.__get_time_plot(self.durations_per_minute),
                 'Histogram of bowel sounds per minute': self.freqs_per_minute,
                 'Histogram of bowel sounds duration': self.durations,
                 'Scatterplot of bowel sounds per minute vs duration': list(zip(self.freqs_per_minute, self.durations_per_minute))
                 }
        return plots

    def __get_freqs_per_minute(self, sounds, time=180):
        freqs_per_minute = int(self.duration / time) * [0]
        i = 0
        for sound in sounds:
            while sound['start'] > (i + 1) * time:
                i += 1
            if i >= int(self.duration / time):
                break
            freqs_per_minute[i] += 1
        if not freqs_per_minute:
            freqs_per_minute = [0]
        freqs_per_minute = np.array(freqs_per_minute).astype(float)
        freqs_per_minute /= time / 60
        return freqs_per_minute

    def __get_durations_per_minute(self, sounds, time=180):
        durations_per_minute = int(self.duration / time) * [0]
        i = 0
        for sound in sounds:
            while sound['start'] > (i + 1) * time:
                i += 1
            if i >= int(self.duration / time):
                break
            durations_per_minute[i] += (sound['end'] - sound['start'])
        if not durations_per_minute:
            durations_per_minute = [0]
        durations_per_minute = np.array(durations_per_minute).astype(float)
        durations_per_minute /= time / 60
        return durations_per_minute

    def __get_durations(self, sounds):
        durations = [1000 * (sound['end'] - sound['start'])
                     for sound in sounds]
        return durations

    def __get_intervals(self, sounds):
        intervals = []
        for i in range(len(sounds) - 1):
            intervals.append(sounds[i + 1]['start'] - sounds[i]['end'])
        return intervals

    def __get_portas_index(self, intervals):
        if not intervals:
            return 0
        b, m = 0, 0
        for i in range(len(intervals) - 1):
            if intervals[i] > intervals[i + 1]:
                b += 1
            if intervals[i] != intervals[i + 1]:
                m += 1
        return b / m * 100

    def __get_guziks_index(self, intervals):
        if not intervals:
            return 0
        sum_l, sum_m = 0.0, 0.0
        for i in range(len(intervals) - 1):
            if intervals[i] < intervals[i + 1]:
                sum_l += np.abs(intervals[i + 1] - intervals[i]) / np.sqrt(2)
            if intervals[i] != intervals[i + 1]:
                sum_m += np.abs(intervals[i + 1] - intervals[i]) / np.sqrt(2)
        return sum_l / sum_m * 100

    def __get_time_plot(self, events_per_minute, time=180):
        times = [time / 60 * (i + 1) for i in range(len(events_per_minute))]
        return list(zip(times, events_per_minute))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str,
                        help='Path to csv file with sounds')
    parser.add_argument('xlsx', type=str,
                        help='Path to save xlsx file')
    parser.add_argument('--wav', type=str, default=None,
                        help='Path to wav file')
    args = parser.parse_args()

    report = Report(args.csv, args.wav)
    report.save(args.xlsx)
    