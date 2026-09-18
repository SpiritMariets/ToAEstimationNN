# Time of arrival estimation of LTE signals for positioning by Neural Networks

<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/SpiritMariets/ToAEstimationNN?color=green)](https://github.com/SpiritMariets/ToAEstimationNN/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/SpiritMariets/ToAEstimationNN)](https://github.com/SpiritMariets/ToAEstimationNN/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/SpiritMariets/ToAEstimationNN.svg?color=0088ff)](https://github.comSpiritMariets/ToAEstimationNN/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/SpiritMariets/ToAEstimationNN.svg?color=7f29d6)](https://github.com/SpiritMariets/ToAEstimationNN/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Zaharov Alexander </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Ilya Sinilshchikov </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Ilya Sinilshchikov </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Разрабатывается нейросетевой метод оценки времени прихода (ToA) LTE-сигнала, превосходящий классические алгоритмы MUSIC и ESPRIT по скорости и точности. Классические методы вычислительно затратны, чувствительны к шуму и многолучёвости, что ограничивает их работу в реальном времени. Предлагается обучить нейронную сеть напрямую отображать корреляционный отклик или частотный канал в оценку ToA. Ожидается снижение RMSE при низком SNR и уменьшение вычислительных затрат на инференсе. Эффективность подтверждается сравнением с MUSIC, ESPRIT и границей Крамера-Рао.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2026}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
