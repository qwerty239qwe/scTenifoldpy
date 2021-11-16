CommandLine tool
-------------

scTenifoldpy also provide command line tools:

config files
====
First, you can generate a config file template by using this command:

.. code-block:: shell

    python -m scTenifold config -t 1 -p ./net_config.yml

Options:
  -t, --type INTEGER RANGE  Type, 1: scTenifoldNet, 2: scTenifoldKnk  [default: 1; 1<=x<=2]
  -p, --path TEXT           Path to generate empty config file  [default: .]
  --help                    Show this message and exit.

Config files contain all the arguments that could be modified and used to run scTenifoldNet or scTenifoldKnk
The only required arguments are X_data_path, Y_data_path (in net), and/or data_path (in knk)

run scTenifoldNet
====

After editing the config file,

.. code-block:: shell

    python -m scTenifold net -c ./net_config.yml -o ./outputs

Options:
  -c, --config TEXT  Loaded config file's path  [required]
  -o, --output TEXT  Output folder containing all analysis results  [default:
                     ./saved_net]
  --help             Show this message and exit.

run scTenifoldKnk
====

.. code-block:: shell

    python -m scTenifold knk -c ./knk_config.yml -o ./outputs

Options:
  -c, --config TEXT  Loaded config file's path  [required]
  -o, --output TEXT  Output folder containing all analysis results  [default:
                     ./saved_net]
  --help             Show this message and exit.
