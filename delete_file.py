import os
import logging


logging.basicConfig(level=logging.INFO)

def delete_file(file_path):
    """
    Удаляет файл по указанному пути.
    
    :param file_path: Путь к файлу, который нужно удалить.
    :return: Строка с результатом операции.
    """
    try:
        os.remove(file_path)
        logging.info(f"Файл {file_path} был успешно удален.")
    except FileNotFoundError:
        logging.info(f"Файл {file_path} не найден.")
    except PermissionError:
        logging.info("Нет прав для удаления файла {file_path}.")
    except Exception as e:
        logging.info("Ошибка при удалении файла {file_path}: {e}")

