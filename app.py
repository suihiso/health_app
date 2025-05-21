import os
import matplotlib
matplotlib.use('Agg')  # Установка неинтерактивного бэкенда
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

# Настройки
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

# Создание приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # ключ для сессии (можно написать любой)

# Проверка допустимого расширения файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Главная страница: загрузка файла
@app.route('/', methods=['GET', 'POST'])
def index():
    # Очистка сессии, если файл уже загружен
    if 'file_path' in session:
        session.pop('file_path')  # удаляем файл из сессии
        session.pop('columns', None)  # удаляем список колонок из сессии
    
    # Очищаем данные о графике при переходе на главную страницу
    if 'chart_file' in session:
        # Удаляем временный файл, если он существует
        chart_file = session.get('chart_file')
        if os.path.exists(chart_file):
            try:
                os.remove(chart_file)
            except Exception:
                pass # Игнорируем ошибки при удалении
        
        # Очищаем данные сессии
        session.pop('chart_file', None)
        session.pop('chart_type', None)
        session.pop('chart_filename', None)

    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Файл не найден", 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash("Файл не выбран", 'danger')  # Используем flash для ошибки
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash("Неподдерживаемый формат файла. Пожалуйста, загрузите файл в формате CSV или Excel.", 'danger')
            return redirect(request.url)

        try:
            # Сохраняем файл
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Читаем файл в pandas
            if file.filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath, engine='openpyxl')

            # Сохраняем путь к файлу и данные в сессии
            session['file_path'] = filepath
            session['columns'] = df.columns.tolist()  # сохраняем список колонок в сессии

            return redirect(url_for('select_action'))  # Переход на страницу с действиями
        
        except Exception as e:
            flash(f"Ошибка при загрузке файла! Текст ошибки: {str(e)}", 'danger')
            return redirect(request.url)

    return render_template('index.html')


# Страница выбора действия
@app.route('/select_action')
def select_action():
    if 'file_path' not in session:
        return redirect(url_for('index'))

    # Очищаем данные о графике при переходе на страницу выбора действия
    if 'chart_file' in session:
        # Удаляем временный файл, если он существует
        chart_file = session.get('chart_file')
        if os.path.exists(chart_file):
            try:
                os.remove(chart_file)
            except Exception:
                pass # Игнорируем ошибки при удалении
        
        # Очищаем данные сессии
        session.pop('chart_file', None)
        session.pop('chart_type', None)
        session.pop('chart_filename', None)

    try:
        # Загружаем данные из сессии
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')
        
        return render_template('select_action.html', columns=df.columns)
    except Exception as e:
        flash(f"Ошибка при чтении файла: {str(e)}", 'danger')
        return redirect(url_for('index'))

# Обзор данных
@app.route('/explore')
def explore():
    if 'file_path' not in session:
        return redirect(url_for('index'))

    # Очищаем данные о графике при переходе на страницу обзора данных
    if 'chart_file' in session:
        # Удаляем временный файл, если он существует
        chart_file = session.get('chart_file')
        if os.path.exists(chart_file):
            try:
                os.remove(chart_file)
            except Exception:
                pass # Игнорируем ошибки при удалении
        
        # Очищаем данные сессии
        session.pop('chart_file', None)
        session.pop('chart_type', None)
        session.pop('chart_filename', None)

    try:
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')

        # Получаем параметр страницы из запроса (по умолчанию 1)
        page = request.args.get('page', 1, type=int)
        rows_per_page = 15
        
        # Определяем общее количество страниц
        total_pages = (len(df) + rows_per_page - 1) // rows_per_page
        
        # Проверяем корректность номера страницы
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages
        
        # Вычисляем индексы начала и конца для выбранной страницы
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        
        # Получаем данные для текущей страницы
        page_data = df.iloc[start_idx:end_idx]
        
        shape = df.shape
        head = page_data.to_html(classes='table table-bordered', index=False)
        
        dtypes = df.dtypes.reset_index()
        dtypes.columns = ['Колонка', 'Тип данных']
        # Пояснения к типам данных
        dtype_explanations = {
            'int64': 'int64 - целое число',
            'float64': 'float64 - вещественное число',
            'object': 'object - строка',
            'bool': 'bool - логическое значение',
            'datetime64[ns]': 'datetime64 - дата/время',
            'category': 'category - категориальные данные'
        }
        dtypes['Тип данных'] = dtypes['Тип данных'].astype(str)
        dtypes['Тип данных'] = dtypes['Тип данных'].apply(lambda x: dtype_explanations.get(x, x + ' - другой тип'))

        nulls = df.isnull().sum().reset_index()
        nulls.columns = ['Колонка', 'Пропущенные значения']

        describe = df.describe(include='all').reset_index()

        # Переименование статистик для пользователя
        translations = {
            'count': 'count - кол-во',
            'mean': 'mean - среднее',
            'std': 'std - стандартное отклонение',
            'min': 'min - минимум',
            '25%': '25% - 1-й квартиль',
            '50%': '50% - медиана',
            '75%': '75% - 3-й квартиль',
            'max': 'max - максимум',
            'unique': 'unique - уникальных значений',
            'top': 'top - наиболее частое значение',
            'freq': 'freq - частота самого частого значения',
            'first': 'first - первая дата',
            'last': 'last - последняя дата'
        }
        
        describe['index'] = describe['index'].apply(lambda x: translations.get(x, x))
        describe = describe.fillna('')
        # Переименовываем заголовок первого столбца
        describe.rename(columns={'index': 'Показатель'}, inplace=True)

        # Анализ аномальных значений
        anomalies = []
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        for column in numeric_columns:
            # Вычисляем квартили и межквартильный размах
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Определяем границы выбросов
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Находим количество выбросов
            outliers_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
            
            if outliers_count > 0:
                anomalies.append({
                    'Колонка': column,
                    'Количество аномалий': outliers_count,
                    'Нижняя граница IQR': round(lower_bound, 2),
                    'Верхняя граница IQR': round(upper_bound, 2)
                })
        
        # Создаем DataFrame с аномалиями и конвертируем в HTML
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_html = anomalies_df.to_html(classes='table table-bordered', index=False)
        else:
            anomalies_html = '<p>Аномальных значений не обнаружено</p>'

        # Информация о пагинации
        pagination = {
            'current_page': page,
            'total_pages': total_pages,
            'has_prev': page > 1,
            'has_next': page < total_pages,
            'is_first_page': page == 1,
            'is_last_page': page == total_pages,
            'rows_showing': f"Строки {start_idx + 1}-{end_idx} из {len(df)}"
        }

        return render_template(
            'explore.html',
            shape=shape,
            head=head,
            dtypes=dtypes.to_html(classes='table table-bordered', index=False),
            nulls=nulls.to_html(classes='table table-bordered', index=False),
            describe=describe.to_html(classes='table table-bordered', index=False),
            anomalies=anomalies_html,
            pagination=pagination
        )
    except Exception as e:
        flash(f"Ошибка при чтении файла: {str(e)}", 'danger')
        return redirect(url_for('select_action'))

# Визуализация данных
@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    # Очищаем возможные сообщения об ошибках, связанные с графиком
    if 'chart_error' in session:
        session.pop('chart_error')

    try:
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')
        columns = df.columns.tolist()
        
        # Значения для повторного отображения выбранных опций
        context = {
            'columns': columns,
            'chart_type': request.form.get('chart_type', 'line'),
            'x_column': request.form.get('x_column'),
            'y_column': request.form.get('y_column')
        }

        if request.method == 'POST':
            try:
                chart_type = request.form['chart_type']
                
                # Для тепловой карты не нужно выбирать переменные
                if chart_type != 'heatmap':
                    x_column = request.form['x_column']
                    
                    # Получаем y_column только если он нужен для данного типа графика
                    y_column = None
                    if chart_type in ['line', 'scatter', 'area']:
                        if 'y_column' not in request.form:
                            return render_template('visualize.html', error="Необходимо выбрать переменную для оси Y", **context)
                        y_column = request.form['y_column']
                else:
                    # Для тепловой карты используем только числовые столбцы
                    numeric_df = df.select_dtypes(include=['float64', 'int64'])
                    if numeric_df.empty or numeric_df.shape[1] < 2:
                        return render_template('visualize.html', 
                                              error="Для построения тепловой карты необходимо минимум 2 числовых столбца в датасете",
                                              **context)
                    x_column = None
                    y_column = None
                
                plt.figure(figsize=(10, 6))
                
                try:
                    # Тепловая карта (особая обработка)
                    if chart_type == 'heatmap':
                        # Вычисляем корреляцию только для числовых столбцов
                        numeric_df = df.select_dtypes(include=['float64', 'int64'])
                        corr = numeric_df.corr()
                        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
                        plt.title('Тепловая карта корреляций числовых переменных')
                    # Линейный график
                    elif chart_type == 'line':
                        plt.plot(df[x_column], df[y_column])
                    # Диаграмма рассеяния
                    elif chart_type == 'scatter':
                        plt.scatter(df[x_column], df[y_column])
                    # Гистограмма
                    elif chart_type == 'hist':
                        plt.hist(df[x_column], bins=20)
                    # Боксплот
                    elif chart_type == 'boxplot':
                        sns.boxplot(x=df[x_column])
                    # Круговая диаграмма
                    elif chart_type == 'pie':
                        df[x_column].value_counts().plot.pie(autopct='%1.1f%%')
                    # Столбчатая диаграмма
                    elif chart_type == 'bar':
                        df[x_column].value_counts().plot.bar()
                    # Диаграмма с областями (Area chart)
                    elif chart_type == 'area':
                        plt.fill_between(df[x_column], df[y_column], color="skyblue", alpha=0.4)
                        plt.plot(df[x_column], df[y_column], color="Slateblue", alpha=0.6)
                    
                    # Настройка заголовка и подписей осей для всех типов кроме тепловой карты
                    if chart_type != 'heatmap':
                        title = f'{chart_type.capitalize()} для {x_column}'
                        if y_column:
                            title += f' и {y_column}'
                        
                        plt.title(title)
                        plt.xlabel('Фактические значения', fontsize=12)
                        plt.ylabel('Предсказанные значения', fontsize=12)
                    
                    # Обеспечиваем чистый буфер
                    plt.tight_layout()
                    
                    # Сохранение графика в буфер
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    # Сохраняем данные в файловую систему вместо сессии
                    chart_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_chart.png')
                    with open(chart_file, 'wb') as f:
                        f.write(buf.getvalue())
                    
                    # Формируем имя файла для скачивания
                    if chart_type == 'heatmap':
                        filename = f"chart_{chart_type}_correlations.png"
                    else:
                        filename = f"chart_{chart_type}_{x_column}.png"
                        if y_column:
                            filename = f"chart_{chart_type}_{x_column}_{y_column}.png"
                    
                    # Сохраняем путь к файлу в сессии
                    session['chart_file'] = chart_file
                    session['chart_type'] = chart_type
                    session['chart_filename'] = filename
                    
                    # Кодируем для отображения на странице
                    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    buf.close()
                    plt.close()
                    
                    # Обновляем context для правильного отображения выбранных опций
                    context.update({
                        'image': image_base64, 
                        'x_column': x_column,
                        'y_column': y_column,
                        'chart_type': chart_type
                    })
                    
                    return render_template('visualize.html', **context)
                
                except Exception as e:
                    plt.close()  # Закрываем фигуру при ошибке
                    return render_template('visualize.html', error=f"Ошибка при построении графика: {str(e)}", **context)
                
            except Exception as e:
                if 'plt' in locals():
                    plt.close()  # Закрываем фигуру при ошибке
                return render_template('visualize.html', error=f"Ошибка: {str(e)}", **context)

        return render_template('visualize.html', **context)
    except Exception as e:
        flash(f"Ошибка при чтении файла: {str(e)}", 'danger')
        return redirect(url_for('index'))

# Скачать график
@app.route('/download_chart', methods=['GET'])
def download_chart():
    # Проверяем наличие файла графика
    model_index = request.args.get('model_index', 0, type=int)
    
    if 'chart_files' not in session or not session['chart_files']:
        flash("График не найден", 'danger')
        return redirect(url_for('training_models'))
    
    chart_files = session['chart_files']
    
    # Проверяем, что индекс модели корректный
    if model_index < 0 or model_index >= len(chart_files):
        flash("Указанный график не найден", 'danger')
        return redirect(url_for('training_models'))
    
    chart_file = chart_files[model_index]
    
    if not os.path.exists(chart_file):
        flash("Файл графика не найден", 'danger')
        return redirect(url_for('training_models'))
    
    # Получаем имя файла для скачивания
    filename = os.path.basename(chart_file)
    
    # Отправляем файл пользователю
    try:
        return send_file(chart_file, as_attachment=True, download_name=filename)
    except Exception as e:
        flash(f"Ошибка при скачивании файла: {str(e)}", 'danger')
        return redirect(url_for('training_models'))

# Метод для получения уникальных значений столбца
@app.route('/get_unique_values')
def get_unique_values():
    if 'file_path' not in session:
        return jsonify({'error': 'Файл не загружен'}), 400

    column = request.args.get('column')
    if not column:
        return jsonify({'error': 'Не указан столбец'}), 400

    try:
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')
        
        # Проверка наличия столбца
        if column not in df.columns:
            return jsonify({'error': 'Указанный столбец не найден'}), 404
        
        # Получение уникальных значений
        unique_values = df[column].dropna().unique().tolist()
        
        # Преобразование в строки для JSON
        unique_values = [str(val) for val in unique_values]
        
        return jsonify({'values': unique_values})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_data', methods=['GET', 'POST'])
def process_data():
    if 'file_path' not in session:
        return redirect(url_for('index'))

    # Очищаем данные о графике при переходе на страницу обработки данных
    if 'chart_file' in session:
        # Удаляем временный файл, если он существует
        chart_file = session.get('chart_file')
        if os.path.exists(chart_file):
            try:
                os.remove(chart_file)
            except Exception:
                pass # Игнорируем ошибки при удалении
        
        # Очищаем данные сессии
        session.pop('chart_file', None)
        session.pop('chart_type', None)
        session.pop('chart_filename', None)

    try:
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')

        # Сохраняем оригинальный датасет, чтобы можно было откатить изменения в случае ошибки
        original_df = df.copy()
        
        # Получаем числовые столбцы для работы с аномальными значениями
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Инициализируем пустой список уникальных значений
        unique_values = []
        
        # Если указан столбец для получения уникальных значений, получаем их
        value_column = request.form.get('value_column')
        if value_column and value_column in df.columns:
            unique_values = df[value_column].dropna().unique().tolist()
            unique_values = [str(val) for val in unique_values]

        if request.method == 'POST':
            action = request.form.get('action')

            # Удалить строки с пропусками в выбранном столбце
            if action == 'drop_na_rows':
                try:
                    na_column = request.form.get('na_column')
                    if not na_column:
                        flash("Не выбран столбец для обработки пропусков", 'danger')
                    else:
                        # Проверяем наличие пропусков
                        if df[na_column].isna().any():
                            original_count = len(df)
                            df = df.dropna(subset=[na_column])
                            removed_count = original_count - len(df)
                            flash(f'Удалено {removed_count} строк с пропусками в столбце "{na_column}"', 'success')
                        else:
                            flash(f'Пропуски в столбце "{na_column}" не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при удалении строк с пропусками: {e}", 'danger')

            # Заполнить пропуски средним значением
            elif action == 'fill_na_mean':
                try:
                    na_column = request.form.get('na_column')
                    if not na_column:
                        flash("Не выбран столбец для обработки пропусков", 'danger')
                    else:
                        if df[na_column].dtype in ['float64', 'int64']:
                            # Проверяем наличие пропусков
                            if df[na_column].isna().any():
                                mean_value = df[na_column].mean()
                                df[na_column].fillna(mean_value, inplace=True)
                                flash(f'Пропуски в столбце "{na_column}" были заполнены средним значением ({mean_value:.2f})', 'success')
                            else:
                                flash(f'Пропуски в столбце "{na_column}" не найдены', 'info')
                        else:
                            flash(f'Столбец "{na_column}" содержит нечисловые данные. Заполнение средним значением невозможно.', 'danger')
                except Exception as e:
                    flash(f"Ошибка при заполнении пропусков средним значением: {e}", 'danger')

            # Заполнить пропуски медианой
            elif action == 'fill_na_median':
                try:
                    na_column = request.form.get('na_column')
                    if not na_column:
                        flash("Не выбран столбец для обработки пропусков", 'danger')
                    else:
                        if df[na_column].dtype in ['float64', 'int64']:
                            # Проверяем наличие пропусков
                            if df[na_column].isna().any():
                                median_value = df[na_column].median()
                                df[na_column].fillna(median_value, inplace=True)
                                flash(f'Пропуски в столбце "{na_column}" были заполнены медианой ({median_value:.2f})', 'success')
                            else:
                                flash(f'Пропуски в столбце "{na_column}" не найдены', 'info')
                        else:
                            flash(f'Столбец "{na_column}" содержит нечисловые данные. Заполнение медианой невозможно.', 'danger')
                except Exception as e:
                    flash(f"Ошибка при заполнении пропусков медианой: {e}", 'danger')

            # Заполнить пропуски модой
            elif action == 'fill_na_mode':
                try:
                    na_column = request.form.get('na_column')
                    if not na_column:
                        flash("Не выбран столбец для обработки пропусков", 'danger')
                    else:
                        # Проверяем наличие пропусков
                        if df[na_column].isna().any():
                            mode_value = df[na_column].mode()[0] if not df[na_column].mode().empty else None
                            if mode_value is not None:
                                df[na_column].fillna(mode_value, inplace=True)
                                flash(f'Пропуски в столбце "{na_column}" были заполнены модой ({mode_value})', 'success')
                            else:
                                flash(f'Не удалось определить моду для столбца "{na_column}"', 'danger')
                        else:
                            flash(f'Пропуски в столбце "{na_column}" не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при заполнении пропусков модой: {e}", 'danger')

            # Заполнить пропуски значением "Неизвестно"
            elif action == 'fill_na_unknown':
                try:
                    na_column = request.form.get('na_column')
                    if not na_column:
                        flash("Не выбран столбец для обработки пропусков", 'danger')
                    else:
                        if df[na_column].dtype not in ['float64', 'int64']:
                            # Проверяем наличие пропусков
                            if df[na_column].isna().any():
                                df[na_column].fillna("Неизвестно", inplace=True)
                                flash(f'Пропуски в столбце "{na_column}" были заполнены значением "Неизвестно"', 'success')
                            else:
                                flash(f'Пропуски в столбце "{na_column}" не найдены', 'info')
                        else:
                            flash(f'Столбец "{na_column}" содержит числовые данные. Заполнение значением "Неизвестно" невозможно.', 'danger')
                except Exception as e:
                    flash(f"Ошибка при заполнении пропусков значением 'Неизвестно': {e}", 'danger')

            # Быстрое удаление всех строк с пропусками
            elif action == 'drop_all_na':
                try:
                    # Проверяем наличие пропусков
                    if df.isna().any().any():
                        original_count = len(df)
                        df.dropna(inplace=True)
                        removed_count = original_count - len(df)
                        flash(f'Удалено {removed_count} строк, содержащих пропуски в любом столбце', 'success')
                    else:
                        flash('Пропуски в данных не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при удалении строк с пропусками: {e}", 'danger')

            # Удалить выбранные столбцы
            elif action == 'drop_columns':
                columns_to_drop = request.form.getlist('columns_to_drop')
                try:
                    df.drop(columns=columns_to_drop, inplace=True)
                    flash(f'Столбцы {", ".join(columns_to_drop)} были удалены', 'success')
                except Exception as e:
                    flash(f"Ошибка при удалении столбцов: {e}", 'danger')

            # Фильтрация по условию
            elif action == 'filter':
                column = request.form['filter_column']
                condition = request.form['condition']
                value = request.form['value']

                try:
                    # Проверяем, что тип данных столбца соответствует введенному значению
                    if df[column].dtype == 'object' and isinstance(value, str):
                        df = df[df[column].str.contains(value, na=False)]
                    elif df[column].dtype in ['int64', 'float64']:
                        df = df.query(f"{column} {condition} {value}")
                    else:
                        flash("Неверный тип данных для фильтрации", 'danger')
                        return redirect(request.url)

                    # Проверка, не стал ли датасет пустым после фильтрации
                    if df.empty:
                        # Если датасет пуст, отменяем фильтрацию и возвращаем исходный датасет
                        df = original_df.copy()
                        flash("Фильтрация не применена, так как результат был пустым.", 'danger')
                    else:
                        flash(f'Данные отфильтрованы по условию: {column} {condition} {value}', 'success')

                except Exception as e:
                    flash(f"Ошибка при фильтрации данных: {e}", 'danger')
                
            # Удаление строк с выбросами
            elif action == 'remove_outlier_rows':
                try:
                    outlier_column = request.form.get('outlier_column')
                    
                    if not outlier_column or outlier_column not in numeric_columns:
                        flash("Не выбран числовой столбец для удаления выбросов", 'danger')
                    else:
                        # Определяем выбросы с помощью межквартильного размаха (IQR)
                        Q1 = df[outlier_column].quantile(0.25)
                        Q3 = df[outlier_column].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Определяем границы выбросов
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Сохраняем количество строк до фильтрации
                        original_count = len(df)
                        
                        # Удаляем выбросы
                        df = df[(df[outlier_column] >= lower_bound) & (df[outlier_column] <= upper_bound)]
                        
                        # Количество удаленных строк
                        removed_count = original_count - len(df)
                        
                        if removed_count > 0:
                            flash(f'Удалено {removed_count} строк с выбросами в столбце "{outlier_column}"', 'success')
                        else:
                            flash(f'Выбросы в столбце "{outlier_column}" не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при удалении выбросов: {e}", 'danger')

            # Замена выбросов на среднее значение
            elif action == 'replace_outliers_mean':
                try:
                    outlier_column = request.form.get('outlier_column')
                    
                    if not outlier_column or outlier_column not in numeric_columns:
                        flash("Не выбран числовой столбец для замены выбросов", 'danger')
                    else:
                        # Определяем выбросы с помощью межквартильного размаха (IQR)
                        Q1 = df[outlier_column].quantile(0.25)
                        Q3 = df[outlier_column].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Определяем границы выбросов
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Находим выбросы
                        outliers = df[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound)]
                        outliers_count = len(outliers)
                        
                        if outliers_count > 0:
                            # Вычисляем среднее значение для замены выбросов
                            mean_value = df[outlier_column].mean()
                            
                            # Заменяем выбросы на среднее
                            df.loc[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound), outlier_column] = mean_value
                            
                            flash(f'Заменено {outliers_count} выбросов на среднее значение ({mean_value:.2f}) в столбце "{outlier_column}"', 'success')
                        else:
                            flash(f'Выбросы в столбце "{outlier_column}" не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при замене выбросов: {e}", 'danger')

            # Замена выбросов на медиану
            elif action == 'replace_outliers_median':
                try:
                    outlier_column = request.form.get('outlier_column')
                    
                    if not outlier_column or outlier_column not in numeric_columns:
                        flash("Не выбран числовой столбец для замены выбросов", 'danger')
                    else:
                        # Определяем выбросы с помощью межквартильного размаха (IQR)
                        Q1 = df[outlier_column].quantile(0.25)
                        Q3 = df[outlier_column].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Определяем границы выбросов
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Находим выбросы
                        outliers = df[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound)]
                        outliers_count = len(outliers)
                        
                        if outliers_count > 0:
                            # Вычисляем медиану для замены выбросов
                            median_value = df[outlier_column].median()
                            
                            # Заменяем выбросы на медиану
                            df.loc[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound), outlier_column] = median_value
                            
                            flash(f'Заменено {outliers_count} выбросов на медиану ({median_value:.2f}) в столбце "{outlier_column}"', 'success')
                        else:
                            flash(f'Выбросы в столбце "{outlier_column}" не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при замене выбросов: {e}", 'danger')

            # Замена выбросов на моду
            elif action == 'replace_outliers_mode':
                try:
                    outlier_column = request.form.get('outlier_column')
                    
                    if not outlier_column or outlier_column not in numeric_columns:
                        flash("Не выбран числовой столбец для замены выбросов", 'danger')
                    else:
                        # Определяем выбросы с помощью межквартильного размаха (IQR)
                        Q1 = df[outlier_column].quantile(0.25)
                        Q3 = df[outlier_column].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Определяем границы выбросов
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Находим выбросы
                        outliers = df[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound)]
                        outliers_count = len(outliers)
                        
                        if outliers_count > 0:
                            # Вычисляем моду для замены выбросов
                            mode_value = df[outlier_column].mode()[0]
                            
                            # Заменяем выбросы на моду
                            df.loc[(df[outlier_column] < lower_bound) | (df[outlier_column] > upper_bound), outlier_column] = mode_value
                            
                            flash(f'Заменено {outliers_count} выбросов на моду ({mode_value:.2f}) в столбце "{outlier_column}"', 'success')
                        else:
                            flash(f'Выбросы в столбце "{outlier_column}" не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при замене выбросов: {e}", 'danger')

            # Быстрое удаление всех строк с выбросами
            elif action == 'remove_all_outliers':
                try:
                    # Проверяем наличие числовых столбцов
                    if not numeric_columns:
                        flash("В датасете нет числовых столбцов для обработки выбросов", 'danger')
                    else:
                        # Сохраняем количество строк до фильтрации
                        original_count = len(df)
                        
                        # Для каждого числового столбца определяем и удаляем выбросы
                        for column in numeric_columns:
                            Q1 = df[column].quantile(0.25)
                            Q3 = df[column].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            # Удаляем выбросы
                            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                        
                        # Количество удаленных строк
                        removed_count = original_count - len(df)
                        
                        if removed_count > 0:
                            flash(f'Удалено {removed_count} строк, содержащих выбросы в любом числовом столбце', 'success')
                        else:
                            flash('Выбросы в числовых столбцах не найдены', 'info')
                except Exception as e:
                    flash(f"Ошибка при удалении выбросов: {e}", 'danger')

            # Сохраняем изменения
            try:
                if session['file_path'].endswith('.csv'):
                    df.to_csv(session['file_path'], index=False)
                else:
                    df.to_excel(session['file_path'], index=False, engine='openpyxl')
            except Exception as e:
                flash(f"Ошибка при сохранении файла: {e}", 'danger')

            return redirect(url_for('process_data'))

        # Обновляем список столбцов, так как они могут измениться после обработки
        columns = df.columns.tolist()

        return render_template('process_data.html', columns=columns, numeric_columns=numeric_columns, unique_values=unique_values)

    except Exception as e:
        flash(f"Ошибка при обработке данных: {str(e)}", 'danger')
        return redirect(url_for('index'))

# Перенаправление со старого URL на новый
@app.route('/clean_data', methods=['GET', 'POST'])
def clean_data():
    return redirect(url_for('process_data'))

# Скачать очищенные данные
@app.route('/download_cleaned_data', methods=['GET'])
def download_cleaned_data():
    if 'file_path' not in session:
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')
    except Exception as e:
        flash(f"Ошибка при чтении файла: {str(e)}", 'danger')
        return redirect(url_for('process_data'))

    # Создаём CSV в памяти
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Отправляем файл как CSV
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name="cleaned_data.csv")

# Прогнозирование
@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    # Параметры пагинации
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Количество строк на странице
    
    # Переменные для хранения сообщений об ошибках
    model_error = None
    data_error = None
    
    # Базовый контекст
    context = {
        'metadata_info': 'Внимание! Для прогнозирования необходимо использовать модель с метаданными, созданную в разделе "Обучение модели прогнозирования".'
    }
    
    if request.method == 'POST' and request.form.get('submit_action') == 'predict':
        # Проверка наличия загруженных файлов
        model_file = request.files.get('model_file')
        test_data_file = request.files.get('test_data_file')
        
        # Проверяем наличие файлов в текущем запросе и в сессии
        has_model = (model_file and model_file.filename != '') or 'model_path' in session
        has_data = (test_data_file and test_data_file.filename != '') or 'test_data_path' in session
        
        # Если нет ни модели, ни данных, показываем обе ошибки
        if not has_model and not has_data:
            model_error = 'Необходимо загрузить модель прогнозирования'
            data_error = 'Необходимо загрузить инференсные данные'
            context['model_error'] = model_error
            context['data_error'] = data_error
            return render_template('forecasting.html', **context)
        
        # Если нет только модели, показываем ошибку только для модели
        if not has_model:
            model_error = 'Необходимо загрузить модель прогнозирования'
            context['model_error'] = model_error
            return render_template('forecasting.html', **context)
        
        # Если нет только данных, показываем ошибку только для данных
        if not has_data:
            data_error = 'Необходимо загрузить инференсные данные'
            context['data_error'] = data_error
            return render_template('forecasting.html', **context)
        
        # Флаги для отслеживания необходимости загрузки новых файлов
        need_to_load_model = model_file and model_file.filename != ''
        need_to_load_data = test_data_file and test_data_file.filename != ''
        
        # Путь к модели (новый или существующий)
        model_path = None
        test_data_path = None
        
        # Обработка загрузки модели
        if need_to_load_model:
            if not model_file.filename.endswith('.joblib'):
                model_error = 'Неподдерживаемый формат файла. Пожалуйста, загрузите файл в формате JOBLIB.'
                context['model_error'] = model_error
                return render_template('forecasting.html', **context)
            
            try:
                # Сохраняем файл модели
                model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
                model_file.save(model_filepath)
                
                # Проверяем, что это действительно модель
                try:
                    model = joblib.load(model_filepath)
                    # Проверяем, что модель имеет необходимые методы (например, predict)
                    # Сначала проверяем, является ли это новым форматом с метаданными
                    if isinstance(model, dict) and 'model' in model and 'feature_columns' in model and 'target_column' in model:
                        # Проверяем, что модель внутри словаря имеет метод predict
                        if not hasattr(model['model'], 'predict'):
                            raise ValueError('Загруженный файл не содержит модель прогнозирования')
                    # Проверяем стандартный формат модели
                    elif not hasattr(model, 'predict'):
                        raise ValueError('Загруженный файл не содержит модель прогнозирования')
                    
                    # Если есть предыдущая модель, удаляем её
                    if 'model_path' in session and os.path.exists(session['model_path']) and session['model_path'] != model_filepath:
                        try:
                            os.remove(session['model_path'])
                        except Exception:
                            pass
                    
                    # Сохраняем путь к файлу модели в сессии
                    session['model_path'] = model_filepath
                    model_path = model_filepath
                    
                except Exception as e:
                    # Если возникла ошибка при загрузке модели, удаляем файл
                    if os.path.exists(model_filepath):
                        os.remove(model_filepath)
                    model_error = f'Ошибка при загрузке модели: {str(e)}'
                    context['model_error'] = model_error
                    return render_template('forecasting.html', **context)
            except Exception as e:
                model_error = f'Ошибка при сохранении файла модели: {str(e)}'
                context['model_error'] = model_error
                return render_template('forecasting.html', **context)
        elif 'model_path' in session:
            # Используем ранее загруженную модель
            model_path = session['model_path']
        else:
            # Модель не загружена
            model_error = 'Необходимо загрузить модель прогнозирования'
            context['model_error'] = model_error
            return render_template('forecasting.html', **context)
        
        # Обработка загрузки тестовых данных
        if need_to_load_data:
            if not allowed_file(test_data_file.filename):
                data_error = 'Неподдерживаемый формат файла. Пожалуйста, загрузите файл в формате CSV или Excel.'
                context['data_error'] = data_error
                return render_template('forecasting.html', **context)
            
            try:
                # Сохраняем файл с тестовыми данными
                test_data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], test_data_file.filename)
                test_data_file.save(test_data_filepath)
                
                # Проверяем, что файл можно прочитать как датасет
                try:
                    if test_data_file.filename.endswith('.csv'):
                        df = pd.read_csv(test_data_filepath)
                    else:
                        df = pd.read_excel(test_data_filepath, engine='openpyxl')
                    
                    # Если есть предыдущий файл с тестовыми данными, удаляем его
                    if 'test_data_path' in session and os.path.exists(session['test_data_path']) and session['test_data_path'] != test_data_filepath:
                        try:
                            os.remove(session['test_data_path'])
                        except Exception:
                            pass
                    
                    # Сохраняем путь к файлу с тестовыми данными в сессии
                    session['test_data_path'] = test_data_filepath
                    test_data_path = test_data_filepath
                    
                except Exception as e:
                    # Если возникла ошибка при чтении данных, удаляем файл
                    if os.path.exists(test_data_filepath):
                        os.remove(test_data_filepath)
                    data_error = f'Ошибка при чтении файла с инференсными данными: {str(e)}'
                    context['data_error'] = data_error
                    return render_template('forecasting.html', **context)
            except Exception as e:
                data_error = f'Ошибка при сохранении файла с инференсными данными: {str(e)}'
                context['data_error'] = data_error
                return render_template('forecasting.html', **context)
        elif 'test_data_path' in session:
            # Используем ранее загруженные данные
            test_data_path = session['test_data_path']
        else:
            # Данные не загружены
            data_error = 'Необходимо загрузить инференсные данные'
            context['data_error'] = data_error
            return render_template('forecasting.html', **context)
        
        # Теперь выполняем прогнозирование
        try:
            # Загружаем модель
            model = joblib.load(model_path)
            
            # Проверяем, содержит ли модель метаданные (новый формат)
            model_data = None
            if isinstance(model, dict) and 'model' in model and 'feature_columns' in model and 'target_column' in model:
                model_data = model
                model = model_data['model']
                required_features = model_data['feature_columns']
            
            # Загружаем данные
            if test_data_path.endswith('.csv'):
                df = pd.read_csv(test_data_path)
            else:
                df = pd.read_excel(test_data_path, engine='openpyxl')
            
            # Определяем требуемые признаки для модели
            required_features = []
            
            # Если у нас уже есть информация о признаках из метаданных
            if model_data and model_data['feature_columns']:
                required_features = model_data['feature_columns']
            # Иначе пытаемся получить информацию из самой модели
            elif hasattr(model, 'feature_names_in_'):
                # Для sklearn >= 1.0
                required_features = model.feature_names_in_.tolist()
            elif hasattr(model, '_Booster') and hasattr(model._Booster, 'feature_names'):
                # Для XGBoost
                required_features = model._Booster.feature_names
            elif hasattr(model, 'feature_importances_') and hasattr(model, 'n_features_in_'):
                # Определяем признаки из обучающих данных, если они доступны
                if 'training_features' in session:
                    required_features = session['training_features']
            
            # Если список признаков не удалось получить, пробуем определить из самой модели
            # или использовать все столбцы из датафрейма в качестве признаков
            if not required_features:
                try:
                    # Пробуем получить признаки из различных атрибутов модели
                    if hasattr(model, 'feature_names_'):
                        required_features = model.feature_names_
                    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                        required_features = model.get_booster().feature_names
                    elif hasattr(model, 'booster') and hasattr(model.booster, 'feature_name'):
                        required_features = model.booster.feature_name()
                    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                        # Для ансамблевых моделей
                        if hasattr(model.estimators_[0], 'feature_names_in_'):
                            required_features = model.estimators_[0].feature_names_in_.tolist()
                except:
                    # В случае ошибки используем все столбцы
                    pass
            
            # Если у нас есть список требуемых признаков
            if required_features:
                # Проверяем наличие всех необходимых признаков в датасете
                available_features = set(df.columns)
                missing_features = [feature for feature in required_features if feature not in available_features]
                
                if missing_features:
                    # Если отсутствуют требуемые признаки, сообщаем об этом
                    data_error = f'В инференсных данных отсутствуют необходимые признаки: {", ".join(missing_features)}'
                    context['data_error'] = data_error
                    return render_template('forecasting.html', **context)
                
                # Подготавливаем датафрейм для прогнозирования, сохраняя правильный порядок признаков
                X_inference = df[required_features].copy()
                
                # Преобразуем категориальные признаки, если они есть
                for column in X_inference.columns:
                    if X_inference[column].dtype == 'object':
                        try:
                            X_inference[column] = pd.to_numeric(X_inference[column], errors='ignore')
                        except:
                            pass
                
                # Сохраняем копию исходных ненормализованных данных
                X_inference_original = X_inference.copy()
                
                # Проверяем, нужно ли нормализовать данные (если модель обучалась на нормализованных данных)
                if model_data.get('normalized', False):
                    # Проверяем, есть ли в метаданных информация о scaler
                    if 'scaler' in model_data and model_data['scaler'] is not None:
                        # Используем сохраненный scaler для нормализации новых данных
                        X_inference = model_data['scaler'].transform(X_inference)
                    else:
                        # Если scaler не сохранен, но данные нормализовались при обучении,
                        # нормализуем новые данные с нуля
                        scaler = StandardScaler()
                        X_inference = scaler.fit_transform(X_inference)
                
                # Делаем прогноз
                try:
                    y_pred = model.predict(X_inference)
                except Exception as e:
                    # Если не получается сделать прогноз, возвращаем ошибку
                    data_error = f'Ошибка при прогнозировании. Данные не подходят для модели: {str(e)}'
                    context['data_error'] = data_error
                    return render_template('forecasting.html', **context)
            
            # Создаем новый датафрейм с исходными ненормализованными признаками
            result_df = X_inference_original.copy()
            
            # Добавляем прогнозы в датафрейм
            result_df['Prediction'] = y_pred
            
            # Сохраняем датафрейм с прогнозами для последующего использования
            session['prediction_df'] = result_df.to_json()
            
            # Определяем тип данных прогноза
            is_classification = model_data.get('task_type') == 'classification'
            prediction_type = 'categorical' if is_classification else 'numeric'
            
            # Сохраняем тип прогноза для отображения соответствующей гистограммы
            session['prediction_type'] = prediction_type
            
            # Перенаправляем на страницу результатов
            return redirect(url_for('forecasting'))
            
        except Exception as e:
            flash(f'Ошибка при выполнении прогнозирования: {str(e)}', 'danger')
            return redirect(url_for('forecasting'))
    
    # Если есть данные прогнозов, отображаем их
    if 'prediction_df' in session:
        try:
            # Загружаем датафрейм с прогнозами (только нужные признаки и прогнозы)
            prediction_df = pd.read_json(session['prediction_df'])
            
            # Получаем тип прогноза
            prediction_type = session.get('prediction_type', 'numeric')
            
            # Общее количество записей
            prediction_count = len(prediction_df)
            
            # Вычисляем общее количество страниц
            total_pages = (prediction_count + per_page - 1) // per_page
            
            # Корректируем текущую страницу, если она выходит за пределы
            if page < 1:
                page = 1
            elif page > total_pages and total_pages > 0:
                page = total_pages
            
            # Выбираем записи для текущей страницы
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, prediction_count)
            
            # Получаем статистику по прогнозам
            if prediction_type == 'numeric':
                prediction_mean = round(prediction_df['Prediction'].mean(), 2)
                prediction_min = round(prediction_df['Prediction'].min(), 2)
                prediction_max = round(prediction_df['Prediction'].max(), 2)
            else:
                # Для категориальных данных вместо числовых характеристик показываем текстовые
                most_common = prediction_df['Prediction'].value_counts().index[0]
                prediction_mean = f"Мода: {most_common}"
                prediction_min = "N/A"
                prediction_max = "N/A"
            
            # Формируем таблицу для отображения
            page_df = prediction_df.iloc[start_idx:end_idx]
            predictions_table = page_df.to_html(classes='table table-striped table-bordered', index=False, justify='left')
            
            # Строим гистограмму в зависимости от типа данных
            plt.figure(figsize=(10, 6))
            
            
            if prediction_type == 'numeric':
                # Гистограмма для числовых данных
                sns.histplot(prediction_df['Prediction'], kde=True)
                plt.title('Распределение прогнозируемых значений')
                plt.xlabel('Значение')
                plt.ylabel('Частота')
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                # Гистограмма для категориальных данных
                category_counts = prediction_df['Prediction'].value_counts()
                sns.barplot(x=category_counts.index, y=category_counts.values)
                plt.title('Распределение прогнозируемых категорий')
                plt.xlabel('Категория')
                plt.ylabel('Количество')
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Сохраняем гистограмму в base64 для отображения
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            hist_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # Формируем таблицу статистик
            describe = prediction_df.describe(include='all').reset_index()
            translations = {
                'count': 'count - кол-во',
                'mean': 'mean - среднее',
                'std': 'std - стандартное отклонение',
                'min': 'min - минимум',
                '25%': '25% - 1-й квартиль',
                '50%': '50% - медиана',
                '75%': '75% - 3-й квартиль',
                'max': 'max - максимум',
                'unique': 'unique - уникальных значений',
                'top': 'top - наиболее частое значение',
                'freq': 'freq - частота самого частого значения',
                'first': 'first - первая дата',
                'last': 'last - последняя дата'
            }
            describe['index'] = describe['index'].apply(lambda x: translations.get(x, x))
            describe = describe.fillna('')
            describe.rename(columns={'index': 'Показатель'}, inplace=True)
            stats_table = describe.to_html(classes='table table-striped table-bordered', index=False)
            
            # Добавляем результаты в контекст
            context.update({
                'predictions': True,  # Флаг наличия прогнозов
                'predictions_table': predictions_table,
                'stats_table': stats_table,
                'hist_image': hist_image,
                'prediction_count': prediction_count,
                'prediction_mean': prediction_mean,
                'prediction_min': prediction_min,
                'prediction_max': prediction_max,
                'current_page': page,
                'total_pages': total_pages
            })
        
        except Exception as e:
            flash(f'Ошибка при отображении результатов прогнозирования: {str(e)}', 'danger')
            # Если возникла ошибка, удаляем данные прогнозов
            session.pop('prediction_df', None)
            session.pop('prediction_type', None)
    
    return render_template('forecasting.html', **context)

@app.route('/training_models', methods=['GET', 'POST'])
def training_models():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    # Очищаем данные о графике при переходе на страницу прогнозирования
    if 'chart_file' in session:
        # Удаляем временный файл, если он существует
        chart_file = session.get('chart_file')
        if os.path.exists(chart_file):
            try:
                os.remove(chart_file)
            except Exception:
                pass # Игнорируем ошибки при удалении
        
        # Очищаем данные сессии
        session.pop('chart_file', None)
        session.pop('chart_type', None)
        session.pop('chart_filename', None)
    
    # Очищаем файлы моделей при каждом переходе на страницу
    if 'model_files' in session:
        model_files = session.get('model_files', [])
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    os.remove(model_file)
                except Exception:
                    pass
        session.pop('model_files', None)

    try:
        df = pd.read_csv(session['file_path']) if session['file_path'].endswith('.csv') else pd.read_excel(session['file_path'], engine='openpyxl')
        
        # Получаем все столбцы
        all_columns = df.columns.tolist()
        
        # Получаем числовые столбцы для регрессии
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Получаем категориальные столбцы для классификации
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Значения для повторного отображения выбранных опций
        context = {
            'all_columns': all_columns,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'task_type': request.form.get('task_type'),  # Убираем значение по умолчанию для открытия начальной инструкции
            'target_column': request.form.get('target_column'),
            'feature_columns': request.form.getlist('feature_columns'),
            'model_types': request.form.getlist('model_types[]')
        }
        
        if request.method == 'POST':
            try:
                task_type = request.form.get('task_type')
                target_column = request.form.get('target_column')
                feature_columns = request.form.getlist('feature_columns')
                model_types = request.form.getlist('model_types[]')
                
                # Проверка ввода
                if not target_column:
                    return render_template('training_models.html', error="Необходимо выбрать целевую переменную", **context)
                
                if not feature_columns:
                    return render_template('training_models.html', error="Необходимо выбрать признаки для прогнозирования", **context)
                
                if not model_types:
                    return render_template('training_models.html', error="Необходимо выбрать хотя бы одну модель прогнозирования", **context)
                
                # Подготовка данных
                X = df[feature_columns]
                y = df[target_column]
                
                # Сохраняем список признаков для будущего использования при прогнозировании
                session['training_features'] = feature_columns
                session['target_column'] = target_column
                
                # Проверка наличия пропущенных значений
                if X.isnull().values.any() or y.isnull().values.any():
                    return render_template('training_models.html', error="В данных есть пропущенные значения. Пожалуйста, обработайте их перед прогнозированием.", **context)
                
                # Предобработка категориальных данных
                X_processed = X.copy()
                encoders = {}
                
                for column in X.columns:
                    if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                        encoder = LabelEncoder()
                        X_processed[column] = encoder.fit_transform(X[column])
                        encoders[column] = encoder
                
                # Нормализация данных
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_processed)
                
                # Разделение на обучающую и тестовую выборки
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
                
                # Список для хранения результатов каждой модели
                results = []
                chart_files = []
                model_files = []
                
                # Обучение моделей
                for model_type in model_types:
                    if task_type == 'regression':
                        if model_type == 'linear':
                            model = LinearRegression()
                            model_name = "Линейная регрессия"
                        elif model_type == 'decision_tree':
                            model = DecisionTreeRegressor(random_state=42)
                            model_name = "Дерево решений (регрессия)"
                        elif model_type == 'random_forest':
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model_name = "Случайный лес (регрессия)"
                        elif model_type == 'gradient_boosting':
                            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                            model_name = "Градиентный бустинг (регрессия)"
                        elif model_type == 'svr':
                            model = SVR()
                            model_name = "Метод опорных векторов (регрессия)"
                        
                        model.fit(X_train, y_train)
                        
                        # Предсказание
                        y_pred = model.predict(X_test)
                        
                        # Метрики
                        metrics = {
                            'MAE': round(mean_absolute_error(y_test, y_pred), 3),
                            'MSE': round(mean_squared_error(y_test, y_pred), 3),
                            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
                            'R2': round(r2_score(y_test, y_pred), 3)
                        }
                        
                        # Автоматический вывод
                        if metrics['R2'] > 0.8:
                            conclusion = f"Модель {model_name} очень хорошо объясняет данные (R² = {metrics['R2']})."
                        elif metrics['R2'] > 0.6:
                            conclusion = f"Модель {model_name} показывает хорошие результаты (R² = {metrics['R2']})."
                        elif metrics['R2'] > 0.4:
                            conclusion = f"Модель {model_name} даёт среднее качество предсказаний (R² = {metrics['R2']})."
                        else:
                            conclusion = f"Модель {model_name} плохо объясняет данные (R² = {metrics['R2']}). Рекомендуется улучшить модель."
                        
                        # Построение графика
                        plt.figure(figsize=(10, 6))
                        scatter = plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', label='Предсказания')
                        
                        # Добавляем линию идеального предсказания
                        min_val = min(min(y_test), min(y_pred))
                        max_val = max(max(y_test), max(y_pred))
                        ideal_line = plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
                        
                        plt.xlabel('Фактические значения', fontsize=12)
                        plt.ylabel('Предсказанные значения', fontsize=12)
                        plt.title(f'График предсказаний - {model_name}', fontsize=14)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend(loc='best', fontsize=10)
                        
                        # Устанавливаем равный масштаб для обеих осей
                        ax = plt.gca()
                        ax.set_aspect('equal', adjustable='box')
                        
                        # Добавляем информацию о метриках на график
                        metrics_text = f"R² = {metrics['R2']}\nRMSE = {metrics['RMSE']}\nMAE = {metrics['MAE']}"
                        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
                                    fontsize=10, ha='left', va='top')
                    
                    elif task_type == 'classification':
                        # Кодирование целевой переменной, если она категориальная
                        if y.dtype == 'object' or y.dtype.name == 'category':
                            target_encoder = LabelEncoder()
                            y = target_encoder.fit_transform(y)
                            y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
                        
                        if model_type == 'logistic':
                            model = LogisticRegression(max_iter=1000, random_state=42)
                            model_name = "Логистическая регрессия"
                        elif model_type == 'decision_tree':
                            model = DecisionTreeClassifier(random_state=42)
                            model_name = "Дерево решений (классификация)"
                        elif model_type == 'random_forest':
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model_name = "Случайный лес (классификация)"
                        elif model_type == 'gradient_boosting':
                            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                            model_name = "Градиентный бустинг (классификация)"
                        elif model_type == 'knn':
                            model = KNeighborsClassifier(n_neighbors=5)
                            model_name = "К-ближайших соседей"
                        elif model_type == 'svc':
                            model = SVC(probability=True, random_state=42)
                            model_name = "Метод опорных векторов (классификация)"
                        
                        model.fit(X_train, y_train)
                        
                        # Предсказание
                        y_pred = model.predict(X_test)
                        
                        # Метрики
                        metrics = {
                            'Accuracy': round(accuracy_score(y_test, y_pred), 3),
                            'Precision': round(precision_score(y_test, y_pred, average='weighted'), 3),
                            'Recall': round(recall_score(y_test, y_pred, average='weighted'), 3),
                            'F1': round(f1_score(y_test, y_pred, average='weighted'), 3)
                        }
                        
                        # Автоматический вывод
                        if metrics['Accuracy'] > 0.9:
                            conclusion = f"Модель {model_name} показывает отличные результаты с точностью {metrics['Accuracy']*100:.1f}%."
                        elif metrics['Accuracy'] > 0.7:
                            conclusion = f"Модель {model_name} даёт хорошие результаты с точностью {metrics['Accuracy']*100:.1f}%."
                        elif metrics['Accuracy'] > 0.5:
                            conclusion = f"Модель {model_name} показывает средние результаты с точностью {metrics['Accuracy']*100:.1f}%."
                        else:
                            conclusion = f"Модель {model_name} имеет низкую точность {metrics['Accuracy']*100:.1f}%. Рекомендуется улучшить модель."
                        
                        # Построение матрицы ошибок
                        plt.figure(figsize=(10, 6))
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Улучшенная тепловая карта
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5,
                                    annot_kws={"size": 12}, cbar_kws={"shrink": .8})
                        
                        # Добавляем названия классов, если возможно
                        if hasattr(model, 'classes_') and len(model.classes_) <= 10:
                            class_names = model.classes_
                            tick_marks = np.arange(len(class_names))
                            plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha='right')
                            plt.yticks(tick_marks + 0.5, class_names, rotation=0)
                        
                        plt.xlabel('Предсказанные метки', fontsize=12)
                        plt.ylabel('Истинные метки', fontsize=12)
                        plt.title(f'Матрица ошибок - {model_name}', fontsize=14)
                        
                        # Добавляем информацию о метриках на график
                        metrics_text = f"Accuracy = {metrics['Accuracy']}\nPrecision = {metrics['Precision']}\nRecall = {metrics['Recall']}\nF1 = {metrics['F1']}"
                        plt.annotate(metrics_text, xy=(0.05, 0.05), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
                                    fontsize=10, ha='left', va='bottom')
                    
                    # Сохранение графика
                    plt.tight_layout()
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    
                    # Формируем имя файла для сохранения на сервере
                    chart_file = os.path.join(app.config['UPLOAD_FOLDER'], f'prediction_chart_{model_type}.png')
                    with open(chart_file, 'wb') as f:
                        f.write(buf.getvalue())
                    
                    chart_files.append(chart_file)
                    
                    # Сохраняем модель с помощью joblib
                    model_filename = f'model_{task_type}_{model_type}.joblib'
                    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
                    joblib.dump(model, model_path)
                    model_files.append(model_path)
                    
                    # Кодируем для отображения на странице
                    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    # Добавляем результаты в список
                    results.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'metrics': metrics,
                        'conclusion': conclusion,
                        'image': image_base64
                    })
                    
                    buf.close()
                    plt.close()
                
                # Сохраняем данные в сессии для скачивания графиков и моделей
                session['chart_files'] = chart_files
                session['model_files'] = model_files
                
                # Обновляем контекст
                context.update({
                    'results': results,
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'model_types': model_types,
                    'task_type': task_type
                })
                
                return render_template('training_models.html', **context)
                
            except Exception as e:
                if 'plt' in locals():
                    plt.close()  # Закрываем фигуру при ошибке
                return render_template('training_models.html', error=f"Ошибка при прогнозировании: {str(e)}", **context)
        
        return render_template('training_models.html', **context)
        
    except Exception as e:
        flash(f"Ошибка при чтении файла: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/download_model', methods=['GET'])
def download_model():
    # Получаем индекс модели
    model_index = request.args.get('model_index', 0, type=int)
    
    if 'model_files' not in session or not session['model_files']:
        flash("Модель не найдена", 'danger')
        return redirect(url_for('training_models'))
    
    model_files = session['model_files']
    
    # Проверяем, что индекс модели корректный
    if model_index < 0 or model_index >= len(model_files):
        flash("Указанная модель не найдена", 'danger')
        return redirect(url_for('training_models'))
    
    model_file = model_files[model_index]
    
    if not os.path.exists(model_file):
        flash("Файл модели не найден", 'danger')
        return redirect(url_for('training_models'))
    
    # Получаем данные о признаках и целевой переменной из сессии
    feature_columns = session.get('feature_columns', [])
    target_column = session.get('target_column', '')
    task_type = session.get('task_type', '')
    
    # Загружаем модель из файла
    model = joblib.load(model_file)
    
    # Создаем словарь с метаданными модели
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'task_type': task_type,
        'normalized': True, # Указываем, что данные были нормализованы при обучении
        'scaler': scaler if 'scaler' in locals() else None # Сохраняем scaler, если он есть
    }
    
    # Создаем новый файл с метаданными
    model_metadata_filename = os.path.splitext(model_file)[0] + '_with_metadata.joblib'
    joblib.dump(model_data, model_metadata_filename)
    
    # Получаем имя файла для скачивания
    filename = os.path.basename(model_metadata_filename)
    
    # Отправляем файл пользователю
    try:
        return send_file(model_metadata_filename, as_attachment=True, download_name=filename)
    except Exception as e:
        flash(f"Ошибка при скачивании модели: {str(e)}", 'danger')
        return redirect(url_for('training_models'))

@app.route('/download_predictions')
def download_predictions():
    if 'prediction_df' not in session:
        flash('Нет доступных прогнозов для скачивания', 'warning')
        return redirect(url_for('forecasting'))
    
    try:
        # Загружаем датафрейм с прогнозами
        prediction_df = pd.read_json(session['prediction_df'])
        
        # Создаем временный буфер для записи Excel
        output = BytesIO()
        prediction_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        
        # Формируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'predictions_{timestamp}.xlsx'
        
        # Сохраняем прогнозы обратно в сессию после скачивания
        session['prediction_df'] = prediction_df.to_json()
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        flash(f'Ошибка при скачивании прогнозов: {str(e)}', 'danger')
        return redirect(url_for('forecasting'))

@app.route('/get_original_dataset_name')
def get_original_dataset_name():
    if 'file_path' not in session:
        return jsonify({'error': 'Файл не загружен'}), 400
    
    # Получаем имя файла из пути
    dataset_name = os.path.basename(session['file_path'])
    return jsonify({'dataset_name': dataset_name})

@app.route('/use_predictions')
def use_predictions():
    if 'prediction_df' not in session:
        flash('Нет доступных прогнозов для использования', 'warning')
        return redirect(url_for('forecasting'))
    
    try:
        # Загружаем датафрейм с прогнозами
        prediction_df = pd.read_json(session['prediction_df'])
        
        # Сохраняем датафрейм с прогнозами как новый файл
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'predictions_{timestamp}.csv')
        prediction_df.to_csv(new_file_path, index=False)
        
        # Обновляем путь к файлу в сессии
        session['file_path'] = new_file_path
        
        # Очищаем данные о прогнозах
        session.pop('prediction_df', None)
        session.pop('prediction_type', None)
        
        return redirect(url_for('select_action'))
    except Exception as e:
        flash(f'Ошибка при использовании прогнозов: {str(e)}', 'danger')
        return redirect(url_for('forecasting'))

@app.route('/check_dataset')
def check_dataset():
    has_dataset = 'file_path' in session
    return jsonify({'has_dataset': has_dataset})

@app.route('/clear_predictions', methods=['POST'])
def clear_predictions():
    # Очищаем все данные, связанные с прогнозированием
    session.pop('prediction_df', None)
    session.pop('prediction_type', None)
    session.pop('model_path', None)
    session.pop('test_data_path', None)
    
    # Удаляем временные файлы, если они существуют
    if 'model_path' in session and os.path.exists(session['model_path']):
        try:
            os.remove(session['model_path'])
        except Exception:
            pass
    
    if 'test_data_path' in session and os.path.exists(session['test_data_path']):
        try:
            os.remove(session['test_data_path'])
        except Exception:
            pass
    
    return jsonify({'status': 'success'})

# Запуск сервера
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Проверяем наличие директории templates
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    app.run(debug=True)
