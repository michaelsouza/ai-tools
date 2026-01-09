import re
import shutil
import os

# Caminhos dos arquivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB_FILE = os.path.join(BASE_DIR, 'Paper', 'references.bib')
TEX_FILE = os.path.join(BASE_DIR, 'Paper', 'paper.tex')

def backup_files():
    shutil.copy(BIB_FILE, BIB_FILE + '.bak')
    shutil.copy(TEX_FILE, TEX_FILE + '.bak')
    print("Backups criados (.bak)")

def normalize_key(entry):
    # Extrair autor, ano e título para desambiguação
    author_match = re.search(r'author\s*=\s*{(.+?)}', entry, re.IGNORECASE | re.DOTALL)
    year_match = re.search(r'year\s*=\s*{?(\d{4})}?', entry, re.IGNORECASE)
    
    if not author_match or not year_match:
        return None

    # Processar autores
    authors = author_match.group(1).replace('\n', ' ')
    first_author = authors.split(' and ')[0].strip()
    # Remover caracteres especiais e pegar apenas o sobrenome
    if ',' in first_author:
        surname = first_author.split(',')[0].strip()
    else:
        surname = first_author.split()[-1].strip()
    
    # Limpar sobrenome (remover acentos e caracteres não alfanuméricos simples)
    surname = re.sub(r'[^a-zA-Z]', '', surname)
    
    year = year_match.group(1)
    
    return f"{surname}{year}"

def main():
    backup_files()

    with open(BIB_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Encontrar todas as entradas bibtex
    # Esta regex é simplificada e assume que as entradas começam com @ e terminam com }
    # Bibtex parser completo seria melhor, mas para este caso regex com cuidado funciona
    # Vamos iterar chave por chave velha
    
    # Regex para capturar chaves atuais: @type{key,
    entries = re.findall(r'(@\w+\s*{\s*([^,]+),.*?)(?=\n@|\Z)', content, re.DOTALL)
    
    key_map = {}
    new_keys_count = {}
    
    new_bib_content = content

    print(f"Encontradas {len(entries)} referências.")
    
    # Primeiro passo: gerar todas as chaves base desejadas
    old_to_base = []
    base_counts = {}
    
    for full_entry, old_key in entries:
        base = normalize_key(full_entry)
        if not base:
             old_to_base.append((old_key, None))
             continue
        old_to_base.append((old_key, base))
        base_counts[base] = base_counts.get(base, 0) + 1
        
    # Segundo passo: atribuir sufixos
    current_counts = {}
    final_key_map = {}
    
    for old_key, base in old_to_base:
        if not base:
            continue
            
        old_key_stripped = old_key.strip()
        
        if base_counts[base] > 1:
            # Precisa de sufixo
            idx = current_counts.get(base, 0)
            suffix = chr(ord('a') + idx)
            new_key = f"{base}{suffix}"
            current_counts[base] = idx + 1
        else:
            new_key = base
        
        # Evitar mapear se já for igual
        if old_key_stripped != new_key:
            final_key_map[old_key_stripped] = new_key
        
    # Substituição no BibTeX
    print("Atualizando chaves no arquivo BibTeX...")
    # Ordenar por tamanho da chave decrescente para evitar substituir substrings de outras chaves
    sorted_old_keys = sorted(final_key_map.keys(), key=len, reverse=True)
    
    for old_key in sorted_old_keys:
        new_key = final_key_map[old_key]
        # Regex rigorosa para a definição da chave no BibTeX: @type{old_key,
        # O \s* permite espaços entre { e a chave
        pattern = re.compile(rf'(@\w+\s*{{\s*){re.escape(old_key)}(,)', re.IGNORECASE)
        new_bib_content = pattern.sub(rf'\1{new_key}\2', new_bib_content)
        
    with open(BIB_FILE, 'w', encoding='utf-8') as f:
        f.write(new_bib_content)
        
    print("Arquivo references.bib atualizado.")
    
    # Substituição no TeX
    print("Atualizando citações no arquivo TeX...")
    with open(TEX_FILE, 'r', encoding='utf-8') as f:
        tex_content = f.read()
        
    for old_key in sorted_old_keys:
        new_key = final_key_map[old_key]
        # Regex para \cite{... old_key ...}
        # Lookbehind para garantir que estamos dentro de uma estrutura, ou apenas delimitadores
        # Delimitadores comuns de chaves em tex: { , space 
        # Cuidado com fim da linha ou fechamento }
        
        # Regex:
        # (?<=\{) = precedido por {
        # | (?<=,) = ou precedido por ,
        # seguido de espaços opcionais, a chave, espaços opcionais
        # seguido de , ou }
        
        pattern = re.compile(rf'(?<=[\{{,])\s*{re.escape(old_key)}\s*(?=[,\}}])')
        tex_content = pattern.sub(new_key, tex_content)
        
    with open(TEX_FILE, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    print("Arquivo paper.tex atualizado.")
    print(f"Total de chaves renomeadas: {len(final_key_map)}")
    
    # Opcional: imprimir mapa
    # for old, new in final_key_map.items():
    #    print(f"{old} -> {new}")

if __name__ == "__main__":
    main()
