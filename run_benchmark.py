#!/usr/bin/python
import sys, os
import pathlib
import tempfile
import subprocess
import numpy

MIN_PYTHON = (3, 6)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)

# This script runs the larcv application performance benchmark.
# It runs in several modes:
# - single process scale up (increasing minibatch size)
# - single node weak scale up (Increase ranks per node at fixed LOCAL minibatch size)
# - single node strong scale up (Increase ranks per node at fixed GLOBAL minibatch size)
# - multinode weak scale up (Increase total ranks with fixed RANKS_PER_NODE and fixed LOCAL minibatch_size)

# Configurations are stored in dataclass files with information per machine, etc.

# Each configuration needs to identify several settings, per dataset.
# General meta data about the datasets (location, channels, etc) is stored
# in config/dataset/


systems = ['ThetaGPU']
benchmarks = ['single_process']

import test_configs.systems as system_config


def run_benchmark(mode: str, system : str):

    if mode not in benchmarks:
        raise Exception(f"Unsupported mode {mode} requested.")

    # Pull the configs for this mode:
    if mode == "single_process":
        import test_configs.single_process as job_config
    # else ... import as job_config


    # Fetch the setup script
    setup_script = getattr(system_config, system)

    env_dict = setup_env(setup_script)

    print(job_config)

    # Now, we have the env_dict and job configuration, we're able to run the job launcher

    if mode == "single_process":
        # For each dataset, scale the minibatch size in log2-space
        # Run for warmup iterations, then run for real iterations.
        for dataset in job_config.datasets:
            this_ds_config = getattr(job_config, dataset)
            # How many points?  What power of 2 is the end point?
            start_power = int(numpy.log2(this_ds_config.start_batch_size))
            end_power   = int(numpy.log2(this_ds_config.end_batch_size)) + 1

            powers      = numpy.arange(start_power, end_power)

            batch_sizes = [2**p for p in powers]
            # points = numpy.geomspace(
            #     start = this_ds_config.start_batch_size,
            #     stop  = this_ds_config.end_batch_size+1,
            #     num   = int(npoints),
            #     dtype = int)
            # print(points)

            # Build up the run configuration:
            base_command = ['python', 'exec.py', 'distributed=False']
            base_command += [f'dataset={this_ds_config.dataset_name}',]
            base_command += [f'dataset.output_shape={this_ds_config.output_shape}',]
            base_command += [f'dataset.input_shape={this_ds_config.input_shape}',]
            for run_size in batch_sizes:
                command = base_command.copy()
                command += [f'id={mode}-warmup',]
                command += [f'minibatch_size={run_size}',]
                print(command)
                # Run the command:
                proc = subprocess.Popen(
                    command,
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                    # cwd=None,
                    env=env_dict
                )

                stdout, stderr = proc.communicate()
                stdout = stdout.decode('utf-8')
                stderr = stderr.decode('utf-8')

def setup_env(setup_string: str, return_env=True):
    '''
    Run the setup script in a subprocess and capture the env afterwards.

    If return_env == True, this will return a copy of the environment
    in a dict.  Otherwise, it just sets environment variables directly.
    '''


    #############################################################
    # Create a temporary file and write the necessary commands,
    # then call the file in shell and save the environment
    #############################################################

    env_dict = dict()

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            p = pathlib.Path(tmpdirname) / pathlib.Path("setup.sh")
            print(p)
            with open(p, 'w') as temp:
                # do stuff with temp file
                for comm in setup_string.split("\n"):
                    temp.write(comm + '\n')
                    temp.write('\n')

            # Create a command:
            command = ['bash', '-c',
               'source {0} && echo \"<<<<<DO NOT REMOVE>>>>>\" && env\n'.format(temp.name)]
            print(command)
            # Execute it
            proc = subprocess.Popen(
                command, stdout = subprocess.PIPE,
                stderr = subprocess.PIPE)

            stdout, stderr = proc.communicate()

        # Out here, outside of the context above, the setup script is gone.
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')

        # If the command was successfull, then source the script.
        # If not successfull, print out information and raise and exception
        if proc.returncode != 0:
            print("Error in setup script")
            print("Output:\n{0}".format(stdout))
            print("Error:\n{0}".format(stderr))
            raise SoftwareConfigException()
        else:
            # print(stdout)
            found = False
            for line in stdout.split("\n"):
                if '<<<<<DO NOT REMOVE>>>>>' in line:
                    found = True
                    continue
                if not found:
                    continue
                (key, _, value) = line.partition("=")
                # print line.partition("=")
                env_dict[key] = value
    finally:
        pass

    if return_env:
        return env_dict
    else:
        for key, value in env_dict.items():
            os.environ[key] = value
        return None

if __name__ == "__main__":
    # Get the command line arguments, which are:
    #  - what benchmark to run
    #  - what system configuration to run

    import argparse


    parser = argparse.ArgumentParser(description='Run the larcv Sparse IO Benchmark.')
    parser.add_argument('--system', type=str, choices=systems, required=True,
                        help='Which system will we run on?')
    parser.add_argument('--benchmark', choices=benchmarks, required=True,
                        help='Which benchmark mode will run?')

    args = parser.parse_args()
    print(args)

    run_benchmark(args.benchmark, args.system)
