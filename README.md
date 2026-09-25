# Нейросетевое восстановление параметров радиоканала сетей 5G для задач позиционирования

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

Высокоточное позиционирование объектов в сетях 5G и перспективных стандартах 6G является критически важной задачей для развития автономного транспорта, промышленной автоматизации и умных устройств. Однако точная навигация существенно затруднена из-за распространения радиосигнала в условиях сложной городской застройки, где сигнал многократно отражается от зданий и окружающих препятствий. Целью данной работы является разработка нейросетевого метода для автоматического извлечения задержек и комплексных коэффициентов распространения отраженных лучей с возможностью адаптации к их переменному количеству. Предлагаемый подход использует архитектуру Transformer, которая преобразует измеренный сырой канальный импульсный отклик радиоканала в последовательность характеристик отдельных лучей. Модель обрабатывает сигнал напрямую и способна самостоятельно определять число составляющих компонентов без использования сложной ручной настройки. Для обучения и валидации нейросети применялся специализированный симулятор QuaDRiGa, позволяющий воссоздать реалистичные пространственные условия распространения радиоволн с различной плотностью отражений. Данный подход позволяет полностью отказаться от громоздких вычислительных процедур и предварительной оценки структуры сигнала, характерных для традиционных математических алгоритмов. Сравнительный анализ показал, что разработка превосходит классические алгоритмы по точности и надежности оценки задержек, особенно при высоком уровне шумов и сильном перекрытии сигналов. Кроме того, предложенный алгоритм обеспечивает значительно более высокую скорость вычислений и сохраняет устойчивость при динамическом изменении параметров среды. Практическая ценность исследования заключается в создании быстродействующего и надежного инструмента оценки характеристик радиоканала для интеграции в навигационные модули беспроводных систем нового поколения.

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
