# Salt and pepper

Реализована программная реализация алгоритма исправления дефектов "соль и перец" для детектирования углов на изображении. 

При одинаковых параметрах запуска точки, обнаруженные алгоритмами, совпадают.

## Время работы
| Разрешение    | CUDA, с             | CPU, с      |
|:-------------:|:---------------:                |:-------------:    |
| 384x256       | 0.18          | 0.23  |
| 512x512       | 0.17412          | 0.575795  |
| 320x428     | 0.1941          | 0.32964  |

<b>Вывод:</b> использование CUDA дает прирост даже на небольшой разрешении.
