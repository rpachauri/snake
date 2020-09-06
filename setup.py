from setuptools import setup, find_packages

setup(
  name='snake',
  version='0.0.1',
  description='OpenAI Gym environment for the Snake game',
  install_requires=['gym', 'tensorflow'],  # Dependencies snake needs
  license='MIT',
  packages=find_packages(),
)