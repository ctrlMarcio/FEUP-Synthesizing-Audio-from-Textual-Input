# Synthesizing Soundscapes from Textual Input: Development and Comparison of Generative AI Models - Ponto de situação

Esta tese tem como objetivo estudar e desenvolver modelos generativos de IA para a síntese de áudio, centrando-se especificamente na síntese de paisagens sonoras a partir de texto. A investigação tem como objetivo criar um sistema que produza amostras de áudio de alta qualidade em função de pedidos textuais.

O state-of-the-art está praticamente pronto e foram realizadas experiências iniciais com modelos generativos. No entanto, o desenvolvimento de modelos tem sido um desafio devido aos requisitos significativos de recursos computacionais. Foram tentados vários modelos de teste, mas ainda não foram alcançados resultados significativos.

A metodologia de investigação envolve o estudo e a replicação de arquitecturas conhecidas para a síntese de áudio, bem como a inspiração em modelos de geração de imagem e de conversão de texto em fala. Os modelos são treinados e avaliados usando várias técnicas, incluindo a medição da divergência KL para avaliação da qualidade.

Não há ainda conclusões ou resultados significativos a comunicar. O desenvolvimento de modelos está em curso e é necessária mais experiências para obter resultados significativos.

Neste momento, dois modelos estão a ser desenvolvidos. Um com base numa ideia própria, que usa GANs no espaço latente para gerar dados; outro que se influencia pela arquitetura do Dall-E 2, usando difusão. Este segundo modelo está a ser construído de raiz. Um terceiro modelo que usa transformers (estilo do GPT) será definido teoricamente, enquanto a sua implementação ficará para future work, devido à falta de tempo. Estes modelos estão a ser construidos num dataset reduzido, a expansão para um dataset do estado da arte será também realizada. (Estas informações encontram-se no cronograma anexado).

O principal problema encontrado nesta investigação é a necessidade de recursos computacionais substanciais para treinar e avaliar os modelos generativos. Dado que estes não existem, optimizações adicionais não encontras nos modelos estado da arte são necessárias.

## Estrutura da tese
1. Introdução (Concluído)
2. Estado da Arte (Concluído)
3. Declaração do problema (Em curso)
4. Solução (Pendente - a aguardar resultados)
5. Conclusão (Pendente - a aguardar resultados)
