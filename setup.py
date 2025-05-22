from setuptools import setup, find_packages

setup(
    name='intelligent_control',
    version='0.1',
    packages=find_packages(),  # Automatically includes `sac` and any other packages
    
    # install_requires=[
    #     'numpy',
    #     'torch',
    #     'gymnasium',
    #     'gymnasium-robotics',
    # ],
)


# pip install -e.
# python experiments/main.py