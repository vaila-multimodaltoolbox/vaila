#!/bin/bash

# Verifique se o ffprobe está instalado
if ! command -v ffprobe &> /dev/null
then
    echo "ffprobe could not be found"
    exit
fi

# Verifique se um diretório foi fornecido
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory_path=$1

# Crie um timestamp para o nome do arquivo de saída
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="$directory_path/metadata_$timestamp"

# Crie um diretório para os arquivos JSON
mkdir -p "$output_dir"

# Iterar sobre todos os arquivos de vídeo no diretório fornecido
for video_file in "$directory_path"/*.{mp4,avi,mov,mkv}; do
    if [ -f "$video_file" ]; then
        # Obtenha o nome base do arquivo de vídeo
        base_name=$(basename "$video_file")
        # Substitua a extensão do arquivo por .json
        json_file="${base_name%.*}.json"
        # Execute o ffprobe e salve o resultado em um arquivo JSON
        ffprobe -v quiet -print_format json -show_format -show_streams "$video_file" > "$output_dir/$json_file"
        echo "Metadata for $video_file saved to $output_dir/$json_file"
    fi
done

echo "Metadata extraction completed."
