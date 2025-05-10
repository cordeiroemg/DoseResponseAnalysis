from shutil import copyfile

# Caminhos para copiar o arquivo original para a nova versão modificada
original_path = "/mnt/data/curveFits.py"
modified_path = "/mnt/data/curveFits_with_total_infested.py"

# Copia o arquivo original
copyfile(original_path, modified_path)

modified_path
