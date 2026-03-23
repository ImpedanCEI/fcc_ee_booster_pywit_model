from setuptools import setup
from setuptools.command.install import install
import os


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        print('Cloning FCC-ee booster lattice/model repository...')
        os.system('git clone https://github.com/ImpedanCEI/fcc_ee_booster_pywit_model.git')
        print('Clone completed.')


setup(
    name='fcc_ee_booster_pywit_model',
    version='0.1.0',
    packages=['fcc_ee_booster_pywit_model'],
    url='https://github.com/ImpedanCEI/fcc_ee_booster_pywit_model',
    license='MIT',
    author='Lorenzo Giacomel, Dora Gibellieri, Carlo Zannini, Chiara Antuono',
    author_email='lorenzo.giacomel@cern.ch, dora.gibellieri@cern.ch, carlo.zannini@cern.ch',
    description='Impedance model of the FCC-ee booster',
    cmdclass={'install': PostInstallCommand},
    include_package_data=True,
    package_data={
        'fcc_ee_booster_pywit_model': [
            'data/*',
            'data/resonators/*',
            'data/broadband_resonators/*',
            'data/elliptic_elements/*',
            'data/machine_layouts/*'
        ]
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "cpymad"
    ]
)
