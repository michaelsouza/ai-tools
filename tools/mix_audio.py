#!/usr/bin/env python3
"""
Script para mixar múltiplas faixas de áudio WAV com ajuste de volume individual.
"""

import argparse
from pathlib import Path
from pydub import AudioSegment


def adjust_volume(audio, db_change):
    """
    Ajusta o volume de um segmento de áudio.

    Args:
        audio: AudioSegment a ser ajustado
        db_change: Mudança em decibéis (valores positivos aumentam, negativos diminuem)

    Returns:
        AudioSegment com volume ajustado
    """
    return audio + db_change


def mix_tracks(tracks_config, output_path):
    """
    Mixa múltiplas faixas de áudio com volumes individuais.

    Args:
        tracks_config: Lista de dicionários com 'path' e 'volume_db'
        output_path: Caminho para salvar o arquivo mixado
    """
    print("\n" + "=" * 60)
    print("MIXANDO FAIXAS DE ÁUDIO")
    print("=" * 60)

    mixed = None

    for i, config in enumerate(tracks_config, 1):
        track_path = config['path']
        volume_db = config['volume_db']

        print(f"\n[{i}/{len(tracks_config)}] Carregando: {Path(track_path).name}")
        print(f"    Ajuste de volume: {volume_db:+.1f} dB")

        audio = AudioSegment.from_wav(track_path)

        if volume_db != 0:
            audio = adjust_volume(audio, volume_db)

        if mixed is None:
            mixed = audio
        else:
            mixed = mixed.overlay(audio)

    print("\n" + "-" * 60)
    print(f"Salvando arquivo mixado em: {output_path}")
    mixed.export(output_path, format="wav")
    print(f"✓ Mixagem concluída com sucesso!")
    print(f"  Arquivo criado: {output_path}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Mixa faixas de áudio WAV com ajuste de volume individual',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Mixar com volumes padrão (0 dB):
  python mix_audio.py -o mixed.wav track1.wav track2.wav track3.wav

  # Mixar com ajustes de volume específicos:
  python mix_audio.py -o mixed.wav \\
      track1.wav:0 \\
      track2.wav:-3 \\
      track3.wav:+2 \\
      track4.wav:-5

  # Mixar as faixas do Demucs (consagracao):
  python mix_audio.py -o consagracao_mixed.wav \\
      demucs-20251214T163404Z-3-001/demucs/consagracao/drums.wav:0 \\
      demucs-20251214T163404Z-3-001/demucs/consagracao/bass.wav:-2 \\
      demucs-20251214T163404Z-3-001/demucs/consagracao/other.wav:0 \\
      demucs-20251214T163404Z-3-001/demucs/consagracao/vocals.wav:+3

Notas:
  - Valores positivos aumentam o volume (+3 = +3dB)
  - Valores negativos diminuem o volume (-3 = -3dB)
  - Se não especificar o volume, será usado 0 dB (sem alteração)
        """
    )

    parser.add_argument(
        'tracks',
        nargs='+',
        help='Faixas de áudio no formato: arquivo.wav ou arquivo.wav:volume_db'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Arquivo de saída para a mixagem final'
    )

    args = parser.parse_args()

    # Processar as faixas e volumes
    tracks_config = []
    for track_spec in args.tracks:
        if ':' in track_spec:
            track_path, volume_str = track_spec.rsplit(':', 1)
            try:
                volume_db = float(volume_str)
            except ValueError:
                print(f"ERRO: Volume inválido '{volume_str}' para {track_path}")
                return 1
        else:
            track_path = track_spec
            volume_db = 0.0

        # Verificar se o arquivo existe
        if not Path(track_path).exists():
            print(f"ERRO: Arquivo não encontrado: {track_path}")
            return 1

        tracks_config.append({
            'path': track_path,
            'volume_db': volume_db
        })

    # Executar a mixagem
    try:
        mix_tracks(tracks_config, args.output)
        return 0
    except Exception as e:
        print(f"\nERRO durante a mixagem: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
