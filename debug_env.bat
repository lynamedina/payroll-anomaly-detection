@echo off
echo --- WHERE PYTHON --- > debug_out.txt
where python >> debug_out.txt 2>&1
echo --- WHERE PIP --- >> debug_out.txt 2>&1
where pip >> debug_out.txt 2>&1
echo --- PYTHON VERSION --- >> debug_out.txt 2>&1
python --version >> debug_out.txt 2>&1
echo --- PIP LIST PANDAS --- >> debug_out.txt 2>&1
python -m pip show pandas >> debug_out.txt 2>&1
echo --- DONE --- >> debug_out.txt
