# NST с возможностью наложить несколько стилей одновременно
Моя задача состоит в том, чтобы преобразовать картину неизвестного художника в стиль Шишкина и одновременно сделать её более реальной. Для этого я написал библиотеку. В main.py довольно мало кода, если вы загляните!

Вот данные о картинках, которые подаются в модель:

![картина](https://github.com/Kukushenok/NST/blob/master/my_task/task_presentation.png?raw=true)

А вот и сам результат!

![Вот результат](https://github.com/Kukushenok/NST/blob/master/my_task_output.png?raw=true)

## Гайд по библиотеке (самое главное):
### DataLoader

В инициализаторе указывается путь к папке, где лежит Ваша задача, и девайс, на котором будет происходить обучение (изначально - "cuda"). В задаче должен быть файлик settings.json, где указано следующее:

"image_size" - размер выходного изображения: [щирина (int),высота (int)]. Если картинка квадратная: длина стороны (int)

"content_image" - путь к основному изображению (относительно файла settings.json)

"style_data" - массив из данных о картинках стилей:
- "style_image" - путь к изображению стиля (относительно файла settings.json)
- "weight" - вес стиля (float). Стиль с меньшим весом будет влиять меньше, чем стиль с весом побольше.

Сумма весов стилей должна быть равна 1, но, если Вам так неудобно, то в коде всё равно они приводяться к такому состоянию.

"max_epochs" - максимальное кол-во эпох (int). Необязательное поле.

Так же, можно просмотреть картинки, вызвав у DataLoader функцию PlotData()

### NST

В инициализаторе указывается задача (DataLoader). Он уже использует предобученный "VGG19". Если Вас это не устраивает, тогда нужно заглянуть в pretrained_data.py и саморучно добавить настроики предобученной модели. Так же, только там можно настроить sheduler_gamma и sheduler_step.

Обучение запускается функцией Run(), которая вернёт конечную картинку. Оно останавливается, изначально, автоматически, когда сглаженная функция потерь начинает расти. Так же, обучение можно настроить:

max_steps - то же самое, что и max_epochs. Если задано, то игнорирует max_epochs, который задан у задачи, и присваивает max_steps

style_weight - вес стиля. Если Вам нужно больше отстранится от исходной картинки, сделайте его больше.

content_weight - вес основного изображения. Если Вам нужно больше сохранить исходную картинку, сделайте его больше.

H - коэффициент сглаживания функции потерь, которая останавливает обучение.

non_stop - если Вам нужно убрать автоматическое остановление, задайте этот аргумент как True

output_delay - частота вывода результатов. Если Вам нужно чаще, уменьшите, если нужно реже - увеличте.

### Plotter

Создайте экземпляр и можно вызывать функции!

ShowOneImage(tensor,label), где tensor - pytorch'евский Tensor, label - заголовок

ShowMultipleImages(tensors,labels), где tensors - список из pytorch'евских Tensor'ов, labels - спизок заголовков

SaveImage(tensor,path="output.png"), где tensor - pytorch'евский Tensor, path - путь, куда сохранять файл

## Ну.. вот и всё.

Для решения Вашей задачи можно просто написать:
```
import nst.data_loader
import nst.nst
from nst.plotter import Plotter

TASK_NAME = "my_task"

NST = nst.nst.NST(nst.data_loader.DataLoader(TASK_NAME))
Plotter().SaveImage(NST.Run(),TASK_NAME+"_output.png")
```

где TASK_NAME - имя Вашей задачи. Да, я просто чу-чуть поменял main.py!

