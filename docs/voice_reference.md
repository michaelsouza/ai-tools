# Gravar a voz de referência

O `md2audio.py --voice <nome>` narra com a sua voz: o Qwen3-TTS 1.7B-Base imita uma gravação curta
(10–20 s) a partir do áudio e da transcrição exata. Este guia diz o que ler e como gravar;
`tools/voice_ref.py` transforma as gravações na referência usada pelo modelo.

## Como gravar

- Cômodo silencioso e com pouco eco (cortinas, sofá, estante ajudam). Desligue ventilador e ar-condicionado.
- Microfone a um palmo da boca, sempre à mesma distância. Fone com microfone ou celular servem.
- Leia com a voz e o ritmo que você quer ouvir nas narrações, sem teatralizar. A clonagem copia o jeito
  de falar da gravação: se você ler devagar, as narrações sairão devagar.
- Deixe um segundo de silêncio antes de começar e depois de terminar; o script corta as pontas.
- Se tropeçar, regrave a leitura inteira. A transcrição precisa bater palavra por palavra.
- Grave **duas vezes cada texto**, um arquivo por leitura, com os nomes `voz_a1`, `voz_a2`, `voz_b1`…
  No Windows, o app **Gravador de Som** serve (salva `.m4a`); depois mova os arquivos para `Downloads`.

## Textos

Cada texto dura uns 15 segundos. O script reconhece sozinho qual deles você leu.

### A: narração

> Bom dia! Hoje eu vou ler um trecho com calma, prestando atenção na pontuação. A chuva fina molhou o jardim, e o cheiro de terra subiu pela janela. Você já reparou como o silêncio da manhã muda a nossa voz?

### B: texto acadêmico

> Neste trabalho, apresentamos um método para resolver sistemas lineares de grande porte. Os experimentos numéricos mostram que o tempo de solução diminui em até quarenta por cento, sem perda de precisão. Esses resultados abrem caminho para aplicações em simulação de reservatórios.

### C: conversa

> Que surpresa boa encontrar você aqui! Faz quanto tempo que a gente não conversa? Eu me lembro bem daquela viagem a Fortaleza, do vento forte na praia e do sabor do caju. Prometa que vamos marcar um café em breve.

## Preparar a referência

```bash
python tools/voice_ref.py michael /mnt/c/Users/Michael/Downloads/voz_*.m4a
```

Para cada gravação, o script converte para WAV mono de 24 kHz, corta o silêncio das pontas,
normaliza o volume, estima o ruído de fundo e transcreve com o Whisper para medir o quanto a leitura
bate com o texto. Ele escolhe a melhor gravação e grava `models/voices/michael.wav` e
`models/voices/michael.txt`. A pasta `models/` é ignorada pelo Git, então a sua voz não sai da máquina.
Use `--take` para forçar uma gravação específica.

Depois:

```bash
python tools/md2audio.py artigo.md --language pt-br --voice michael
```
