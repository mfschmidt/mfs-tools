#!/usr/bin/env python3

# plot_memory_usage.py

import sys
import re
from pathlib import Path
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Some globals
def get_env(args):
    """ Integrate environment variables into our args. """

    return args


class App:

    # Processes that may come up, known to be irrelevant to our needs
    ignorable_procs = [
        "jetbrains-toolbox",
        "gnome-shell",
        "gnome-system-monitor",
        "dockerd",
        "sshd",
        "xrdp",
        "Xorg",
        "ps -eo pmem,pcpu,vsize,pid,cmd",
    ]

    re_pfm_cmd = re.compile(r"python.*(mfs_make_infomap_networks).*run-([A-Za-z0-9]+)_space")
    re_infomap_cmd = re.compile(r"python.*(infomap).*sm-([0-9.]+)_dens-([0-9.]+)_pajek")
    re_fs_cmd = re.compile(r"(mri[A-Za-z0-9_-]+) (.*)")
    re_ants_cmd = re.compile(r"(Ants[A-Za-z0-9_-]+) (.*)")
    re_docker_cmd = re.compile(r"python.*run.py /output/_dockerized_(.*).inp")

    def __init__(self):
        # Make a place to log errors, wherever they arise
        self.errors = list()
        self.warnings = list()
        self.valid = True

        # Handle command-line arguments, combined with environment
        self.args = self.get_arguments()
        self.args = get_env(self.args)
        self.validate_args()

        # Only bother with the rest if arguments are valid
        if len(self.errors) == 0:
            # Extract information from the log file provided
            setattr(self.args, "input_path", Path(self.args.mem_log_file).parent)

            path_pattern = re.compile(
                r'sub-([A-Za-z0-9]+)/ses-([A-Za-z0-9]+)/run-([A-Za-z0-9]+)'
            )
            match = path_pattern.search(str(self.args.input_path))
            if match:
                self.subject_id = match.group(1)
                self.session_id = match.group(2)
                self.run_id = match.group(3)
                """
                ssr_str = f"sub-{self.subject_id}_ses-{self.session_id}*run-{self.run_id}"
                self.run_log_file = None
                for log_file in Path(self.args.input_path).glob(f"{ssr_str}_log.txt"):
                    self.run_log_file = log_file
                    break
                if (self.run_log_file is None) or (not self.run_log_file.exists()):
                    self.warnings.append(
                        f"There is no run log file in the input path."
                    )
                """
            else:
                self.subject_id = "unknown"
                self.session_id = "unknown"
                self.run_id = "0"
                self.errors.append(
                    "Subject, session, or path could not be parsed."
                )

        # Report any errors encountered
        if len(self.errors) > 0:
            self.valid = False
            for error in self.errors:
                print(error)
        for warning in self.warnings:
            print(warning)

    def validate_args(self):
        """ Ensure the environment will support the requested workflows. """

        if Path(self.args.mem_log_file).exists():
            setattr(self.args, "mem_log_file", Path(self.args.mem_log_file))
            print(f"Path '{self.args.mem_log_file}' exists.")
        else:
            self.errors.append(
                f"Path '{self.args.mem_log_file}' does not exist."
            )

        if (
                ("pfm_log_file" in self.args) and
                (self.args.pfm_log_file is not None) and
                Path(self.args.pfm_log_file).exists()
        ):
            setattr(self.args, "pfm_log_file", Path(self.args.pfm_log_file))

    def printc(self, s, c=""):
        """ print in color """

        color_dict = {
            'black': "\033[0;30m",
            'dark gray': "\033[1;30m",
            'light gray': "\033[0;37m",
            'white': "\033[1;37m",
            'blue': "\033[0;34m",
            'light blue': "\033[1;34m",
            'purple': "\033[0;35m",
            'light purple': "\033[1;35m",
            'cyan': "\033[0;36m",
            'light cyan': "\033[1;36m",
            'orange': "\033[0;33m",
            'yellow': "\033[1;33m",
            'green': "\033[0;32m",
            'light green': "\033[1;32m",
            'red': "\033[0;31m",
            'light red': "\033[1;31m",
        }
        if self.args.verbose:
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                print(color_dict.get(c.lower(), "") + s + "\033[0m")
            else:
                print(s)

    @staticmethod
    def get_arguments():
        """ Parse command line arguments """
    
        parser = argparse.ArgumentParser(
            description="plot_memory_usage.py will read memory log files "
                        "and plot the time taken for each portion of the "
                        "algorithm alongside the memory consumed.",
        )
        parser.add_argument(
            "mem_log_file",
            help="The path to a memory log file "
                 "probably like '.../pfm_*/sub-*/ses-*/run-*/mem_log.txt'",
        )
        parser.add_argument(
            "-p", "--pfm-log-file",
            help="Supply a PFM log file to plot along with the memory.",
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="set to trigger verbose output",
        )
    
        return parser.parse_args()

    @staticmethod
    def event_from_sys_line(match):
        """ Interpret data as system-level memory usage. """

        event = dict()
        if match.group(3) == "kB":
            divisor = 1024 * 1024
        elif match.group(3) == "MB":
            divisor = 1024
        else:
            divisor = 1024 * 1024 * 1024
        if match.group(1) == "Date":
            dt = match.group(2)
            date_str = f"{dt[0:4]}-{dt[4:6]}-{dt[6:8]}"
            time_str = f"{dt[9:11]}:{dt[11:13]}:{dt[13:15]}"
            event["date"] = " ".join([date_str, time_str])
        elif match.group(1) == "MemTotal":
            event["total_memory"] = float(match.group(2)) / divisor
        elif match.group(1) == "MemFree":
            event["free_memory"] = float(match.group(2)) / divisor
        elif match.group(1) == "MemAvailable":
            event["available_memory"] = float(match.group(2)) / divisor

        return event

    @staticmethod
    def event_from_proc_line(match):
        """ Interpret data as process-level memory usage. """

        event = dict()
        """
        cmd_alias = None
        re_pfm_match = re_pfm_cmd.search(line)
        re_infomap_match = re_infomap_cmd.search(line)
        re_fs_match = re_fs_cmd.search(line)
        re_ants_match = re_ants_cmd.search(line)
        re_docker_match = re_docker_cmd.search(line)
        if re_pfm_match:
            cmd_alias = f"pfm_run-{re_pfm_match.group(2)}"
        if re_infomap_match:
            cmd_alias = f"infomap_sm-{re_infomap_match.group(2)}_dens-{re_infomap_match.group(3)}"
        if re_fs_match:
            cmd_alias = f"{re_fs_match.group(1)}"
        if re_ants_match:
            cmd_alias = f"{re_ants_match.group(1)}"
        if re_docker_match:
            cmd_alias = f"{re_docker_match.group(1)}"
        if cmd_alias is not None:
        """
        event['pct_mem'] = float(match.group(1))
        event['pct_cpu'] = float(match.group(2))
        event['virt_size'] = float(match.group(3)) / (1024 * 1024)
        event['pid'] = str(match.group(4))
        event['cmd'] = match.group(5).split(" ")[0]
        return event

    def read_mem_log(self):
        if not self.valid:
            self.printc("The app state is not valid; I cannot run.", c="red")
            return 1

        # Time it
        start = datetime.datetime.now()

        # A tsv file is OK, too. Whichever is supplied as a logfile, we'll swap accordingly.
        tsv_file = Path(str(self.args.mem_log_file).replace(".txt", ".tsv"))
        if tsv_file.exists():
            self.printc(
                f"Found {tsv_file.name}; loading it instead of parsing the log.",
                c="green"
            )
            df = pd.read_csv(tsv_file, sep='\t')
        else:
            # Read the log file, saving each time point as a row in a dataframe
            self.printc(
                f"Parsing {Path(self.args.mem_log_file).name} for memory samples."
            )
            # Pre-compile the res
            # Every non-"-----..." line is either system memory or process memory:
            re_sys_mem = re.compile(r"([A-Za-z]+):\s+([0-9_]+)\s*([A-Za-z]*)")
            re_proc_mem = re.compile(r"([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s+([0-9]+)\s+(.*)")

            # We could also try to only grab known-relevant processes
            events = list()
            this_event = dict()
            with open(self.args.mem_log_file, "r") as f:
                for i, line in enumerate(f.readlines()):
                    # if self.args.verbose:
                    #     self.printc(f"{i:>7}. {line.strip()}")
                    # Parse the line efficiently, moving on if we find a hit.

                    # If we come across a new sample, reset the dict.
                    re_sys_match, re_proc_match = None, None
                    if "----------" in line:
                        if len(this_event.keys()) > 0:
                            this_event = dict()
                        continue  # after storing the last event and clearing the dict
                    else:
                        re_sys_match = re_sys_mem.search(line)
                        if not re_sys_match:
                            re_proc_match = re_proc_mem.search(line)

                        # If we hit system memory information, store it.
                        if re_sys_match:
                            self.printc(re_sys_match.groups(), c="cyan")
                            this_event.update(self.event_from_sys_line(re_sys_match))

                        # If we hit process memory information, store it.
                        elif re_proc_match:
                            self.printc(re_proc_match.groups(), c="cyan")
                            # Ignore background processes, unrelated to active computing
                            for ignorable_proc in self.ignorable_procs:
                                if ignorable_proc in re_proc_match.group(5):
                                    continue
                            if re_proc_match.group(5).startswith("["):
                                continue
                            this_event.update(
                                self.event_from_proc_line(re_proc_match)
                            )

                            # Every process discovered should get its own record
                            events.append(this_event.copy())
                            if self.args.verbose:
                                for k, v in this_event.items():
                                    self.printc(f"  {k}: {v}", c="cyan")
                        else:
                            self.printc(f"Missed the line, {line}", c="yellow")

                        # For debugging a smaller sample, we can stop early
                        # if i > 200:
                        #     break

            # Put it all into a replacement results file
            df = pd.DataFrame(events).sort_values(['date', ], ascending=True)
            df.to_csv(tsv_file, sep="\t", index=False)

            end = datetime.datetime.now()
            self.printc("Wrote results to {} in {}.".format(
                str(tsv_file),
                end - start
            ))

        print(
            f"  {len(df)} events with {len(df['pid'].unique())} processes "
            f"and {len(df['cmd'].unique())} commands in mem data."
        )
        for proc in df['cmd'].unique():
            print(f"  * {proc}")
        return df

    def read_pfm_log(self):
        # Time it
        start = datetime.datetime.now()

        if self.args.pfm_log_file is None:
            return None

        # A tsv file is OK, too. Whichever is supplied as a logfile, we'll swap accordingly.
        tsv_file = Path(str(self.args.pfm_log_file).replace(".txt", ".tsv"))
        if tsv_file.exists():
            self.printc(
                f"Found {tsv_file.name}; loading it instead of parsing the log.",
                c="green"
            )
            df = pd.read_csv(tsv_file, sep='\t')
        elif self.args.pfm_log_file.exists():
            # Read the log file, saving each time point as a row in a dataframe
            self.printc(
                f"Parsing {str(self.args.pfm_log_file.name)} for run events."
            )
            # Pre-compile the re
            re_event = re.compile(r".*Step ([0-9]+). (.*) at "
                                  r"([0-9]+)-([0-9]+)-([0-9]+)\s"
                                  r"([0-9]+):([0-9]+):([0-9.]+)")
            re_infomap = re.compile(r"Starting infomap with graph density "
                                    r"([0-9.]+) at "
                                    r"([0-9]+)-([0-9]+)-([0-9]+)\s"
                                    r"([0-9]+):([0-9]+):([0-9.]+)\.\.\.")
            re_smooth = re.compile(r"Starting smoothing kernel ([0-9.]+)\.\.\.")
            re_rem_dens = re.compile(r"removing density ([0-9.]+)'s community "
                                     r"([0-9]+) with (.*) members")
            events = list()
            this_event = dict()
            removal_counts = dict()
            cur_desc = ""
            cur_smooth = ""
            with open(self.args.pfm_log_file, "r") as f:
                for i, line in enumerate(f.readlines()):
                    # if self.args.verbose:
                    #     self.printc(f"{i:>7}. {line}")
                    # Parse the line efficiently, moving on if we find a hit.
                    # If we come across a new time point, reset everything.
                    re_event_match = re_event.search(line)

                    # If we hit system memory information, store it.
                    if re_event_match:
                        self.printc(re_event_match.groups(), c="cyan")
                        cur_step = re_event_match.group(1)
                        this_event['step'] = cur_step
                        this_event['desc'] = re_event_match.group(2)
                        date_str = "-".join([
                            f"{re_event_match.group(3)}",
                            f"{re_event_match.group(4)}",
                            f"{re_event_match.group(5)}",
                        ])
                        time_str = ":".join([
                            f"{re_event_match.group(6)}",
                            f"{re_event_match.group(7)}",
                            f"{re_event_match.group(8)}",
                        ])
                        this_event["date"] = " ".join([date_str, time_str])
                        this_event["y"] = 0  # just to help matplotlib place a dot
                        # Every process discovered should get its own record
                        events.append(this_event.copy())
                        cur_desc = re_event_match.group(2)
                        if self.args.verbose:
                            for k, v in this_event.items():
                                self.printc(f"  {k}: {v}", c="cyan")
                    elif re_infomap.search(line):
                        re_infomap_match = re_infomap.search(line)
                        cur_dens = re_infomap_match.group(1)
                        this_event['step'] = cur_step
                        this_event['desc'] = "infomap"
                        date_str = "-".join([
                            f"{re_infomap_match.group(2)}",
                            f"{re_infomap_match.group(3)}",
                            f"{re_infomap_match.group(4)}",
                        ])
                        time_str = ":".join([
                            f"{re_infomap_match.group(5)}",
                            f"{re_infomap_match.group(6)}",
                            f"{re_infomap_match.group(7)}",
                        ])
                        this_event["date"] = " ".join([date_str, time_str])
                        this_event["y"] = 0  # just to help matplotlib place a dot
                        # Every process discovered should get its own record
                        events.append(this_event.copy())
                        cur_desc = re_infomap_match.group(2)
                        if self.args.verbose:
                            for k, v in this_event.items():
                                self.printc(f"  {k}: {v}", c="cyan")
                    elif re_smooth.search(line):
                        re_smooth_match = re_smooth.search(line)
                        cur_smooth = re_smooth_match.group(1)
                    elif re_rem_dens.search(line):
                        # Count removed communities
                        re_rem_dens_match = re_rem_dens.search(line)
                        dens = re_rem_dens_match.group(1)
                        cur_key = f"{cur_desc}_sm-{cur_smooth}_dens-{dens}"
                        if cur_key in removal_counts.keys():
                            removal_counts[cur_key] += 1
                        else:
                            removal_counts[cur_key] = 1
                    else:
                        # We're going to be missing the vast majority of lines,
                        # and we don't need to print them all
                        pass

                    # For debugging a smaller sample, we can stop early
                    # if i > 200:
                    #     break

            # Report removed communities per infomap run
            print(f"Network construction:")
            for k, v in removal_counts.items():
                print(f"  * {k}: removed {v:,} communities")

            # Put it all into a replacement results file
            df = pd.DataFrame(events).sort_values(['date', ], ascending=True)
            df.to_csv(tsv_file, sep="\t", index=False)

            end = datetime.datetime.now()
            self.printc("Wrote results to {} in {}.".format(
                str(tsv_file),
                end - start
            ))
        else:
            print(f"No log file found at {self.args.pfm_log_file}.")
            return None
        print(
            f"  {len(df)} events in {len(df['step'].unique())} steps in run data."
        )
        return df

    def plot(self, run_data, mem_data):
        if run_data is not None:
            run_data['date'] = pd.to_datetime(run_data['date'])
        mem_data['date'] = pd.to_datetime(mem_data['date'])
        mem_data['used'] = mem_data['total_memory'] - mem_data['available_memory']
        fig, axes = plt.subplots(
            nrows=2,
            figsize=(16, 9),
            layout='tight',
        )
        mem_ax = axes[0]
        sns.lineplot(
            data=mem_data,
            x="date",
            y="used",
            color="gray",
            linewidth=3,
            linestyle=":",
            ax=mem_ax,
        )
        sns.lineplot(
            data=mem_data,
            x="date",
            y="virt_size",
            hue="cmd",
            ax=mem_ax,
        )
        if run_data is not None:
            sns.scatterplot(
                data=run_data,
                x="date",
                y="y",
                ax=mem_ax,
            )
        if run_data is None:
            dates = sorted([
                mem_data['date'].min(),
                mem_data['date'].max(),
            ])
        else:
            dates = sorted([
                run_data['date'].min(),
                run_data['date'].max(),
                mem_data['date'].min(),
                mem_data['date'].max(),
            ])
        # Annotate the top peak
        peak_mem = mem_data['used'].max()
        peak_date = mem_data['date'].loc[mem_data['used'].idxmax()]
        date_margin = (mem_data['date'].max() - mem_data['date'].min()) / 20.0
        mem_ax.annotate(
            f"Peak memory usage: {peak_mem:.1f} GB\n",
            xy=(peak_date, peak_mem),
            xytext=(peak_date + date_margin, peak_mem * 0.95),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            ha='left', va='top'
        )
        mem_ax.set_title(f"PFM Memory Usage over time for "
                       f"sub-{self.subject_id} ses-{self.session_id} run-{self.run_id}")
        mem_ax.set_ylabel("Memory (GB)")
        mem_ax.legend(bbox_to_anchor=(1.05, 0.50), loc="center left")
        mem_ax.set_xticks(dates)
        mem_ax.set_xticklabels(
            [d.strftime("%m/%d %H:%M") for d in dates],
            rotation=90
        )

        cpu_ax = axes[1]
        cpu_data = mem_data.groupby('date').sum('pct_cpu')
        sns.lineplot(
            data=cpu_data,
            x="date",
            y="pct_cpu",
            color="gray",
            linewidth=3,
            linestyle=":",
            ax=cpu_ax,
        )
        plot_filename = self.args.mem_log_file.name.replace(".txt", ".png")
        # (f"sub-{self.subject_id}_ses-{self.session_id}_"
        #  f"run-{self.run_id}_memory.png")
        fig.savefig(Path(self.args.input_path) / plot_filename)
        print(f"Wrote figure to {str(self.args.input_path)} as {plot_filename}.")

    def run(self):
        if not self.valid:
            print("The app state is not valid; I can't run.")
            return 1

        mem_data = self.read_mem_log()
        run_data = self.read_pfm_log()
        self.plot(run_data, mem_data)

        return 0


def main():
    """ Entry point """

    app = App()
    return app.run()


if __name__ == "__main__":
    main()
