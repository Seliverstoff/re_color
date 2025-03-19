# Импортируем необходимые библиотеки
import torch  # Библиотека для работы с тензорами и нейронными сетями
from deoldify.visualize import *  # Модуль для раскрашивания изображений
from pathlib import Path  # Модуль для работы с путями к файлам

# Указываем устройство для выполнения вычислений (в данном случае CPU)
device = torch.device("cpu")
print(f"Используемое устройство: {device}")
print(f"Количество CUDA устройств: {torch.cuda.device_count()}")  # Выводим количество доступных GPU

# Указываем директории для входных и выходных файлов
photo_dir = Path(r"D:\Restore\N2_out")  # Директория с исходными фотографиями
output_dir = Path(r"D:\Restore\N2_out_s")  # Директория для сохранения обработанных фотографий
output_dir.mkdir(parents=True, exist_ok=True)  # Создаем директорию, если она не существует

# Перехватываем функцию torch.load для принудительного использования CPU
original_torch_load = torch.load  # Сохраняем оригинальную функцию torch.load
def custom_torch_load(*args, **kwargs):
    kwargs['map_location'] = "cpu"  # Указываем, что загрузка должна происходить на CPU
    return original_torch_load(*args, **kwargs)  # Вызываем оригинальную функцию с новыми параметрами
torch.load = custom_torch_load  # Заменяем оригинальную функцию на нашу кастомную

# Выводим информацию о доступности CUDA перед загрузкой модели
print("Перед загрузкой модели:")
print(f"CUDA доступен: {torch.cuda.is_available()}")  # Проверяем, доступен ли CUDA
print(f"Количество CUDA устройств: {torch.cuda.device_count()}")  # Выводим количество доступных GPU

# Загружаем модель для раскрашивания изображений
colorizer = get_image_colorizer(artistic=True)  # Используем художественную модель для раскрашивания

# Выводим информацию о доступности CUDA после загрузки модели
print("После загрузки модели:")
print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"Количество CUDA устройств: {torch.cuda.device_count()}")

# Создаем множество с именами уже обработанных файлов
processed_files = {file.name for file in output_dir.glob("*")}

# Обрабатываем каждое изображение в директории
for photo_path in photo_dir.glob("*"):
    if photo_path.is_file() and photo_path.name not in processed_files:  # Проверяем, что файл не был обработан
        try:
            print(f"Обработка файла: {photo_path.name}")
            # Раскрашиваем изображение и сохраняем результат
            colorizer.plot_transformed_image(
                path=photo_path,  # Путь к исходному изображению
                render_factor=35,  # Параметр, влияющий на качество раскрашивания
                watermarked=False,  # Отключаем водяной знак
                compare=True,  # Показывать сравнение исходного и обработанного изображения
                results_dir=output_dir  # Директория для сохранения результата
            )
        except Exception as e:
            print(f"Ошибка при обработке файла {photo_path.name}: {e}")  # Выводим ошибку, если что-то пошло не так
    else:
        print(f"Файл уже обработан: {photo_path.name}")  # Сообщаем, что файл уже был обработан