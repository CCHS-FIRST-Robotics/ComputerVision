# Python Environment Setup
## Pyenv setup 
```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```
## Linux Environment variables 
[Mac or Windows](#mac-unix-and-windows-instructions)
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

## Virtual env setup
```
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
```
Make sure .bashrc has the following lines
```
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
## Install python and virtual environment
```
pyenv install 3.11.1
pyenv virtualenv 3.11.1 cchs
```
## Mac, Unix and Windows instructions
[Unix and Mac instructions](https://github.com/pyenv/pyenv)  
[Windows instructions](https://github.com/pyenv-win/pyenv-win)

## Activate environment
```
pyenv activate cchs
```