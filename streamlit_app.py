import dask.dataframe as dd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress


# Wczytanie danych
@st.cache_resource
def load_data():
    flights = dd.read_csv(
        '/Users/krystian/Desktop/flight-delay-analysis/flights_ready.csv',
        assume_missing=True,
    )
    airlines = dd.read_csv('/Users/krystian/Desktop/flight-delay-analysis/airlines.csv')
    airports = dd.read_csv('/Users/krystian/Desktop/flight-delay-analysis/airports.csv')
    return flights, airlines, airports

flights, airlines, airports = load_data()

# Mapowanie linii lotniczych i lotnisk
@st.cache_resource
def get_mappings():
    airline_dict = airlines.set_index('IATA_CODE')['AIRLINE'].compute().to_dict()
    airport_dict = airports.set_index('IATA_CODE')['AIRPORT'].compute().to_dict()
    return airline_dict, airport_dict

airline_dict, airport_dict = get_mappings()

st.title('Analiza Lotów Krajowych w USA')
st.sidebar.title('Filtry')

# Wybór okresu analizy
start_year = st.sidebar.selectbox('Wybierz rok początkowy', range(2009, 2019))
start_month = st.sidebar.selectbox('Wybierz miesiąc początkowy', range(1, 13))
end_year = st.sidebar.selectbox('Wybierz rok końcowy', range(2009, 2019))
end_month = st.sidebar.selectbox('Wybierz miesiąc końcowy', range(1, 13))

# Wybór progu opóźnienia
delay_threshold = st.sidebar.selectbox('Wybierz próg opóźnienia (minuty)', [0, 15], index=1)

if (end_year < start_year) or (end_year == start_year and end_month < start_month):
    st.error("Data końcowa musi być późniejsza niż data początkowa.")
else:
    # Filtr dat
    filtered_flights = flights[
        ((flights['year'] > start_year) | ((flights['year'] == start_year) & (flights['month'] >= start_month))) &
        ((flights['year'] < end_year) | ((flights['year'] == end_year) & (flights['month'] <= end_month)))
    ]

    # Filtr linii lotniczych
    airline_options = list(airline_dict.values())
    selected_airlines = st.sidebar.multiselect(
        'Wybierz linie lotnicze (opcjonalnie)',
        airline_options,
        default=[]
    )

    # Filtr lotniska wylotowego
    origin_airports = list(airport_dict.values())
    selected_origin_airport = st.sidebar.selectbox(
        'Wybierz lotnisko wylotowe (opcjonalnie)',
        ['Wszystkie'] + origin_airports,
        index=0
    )

    # Filtr lotniska docelowego
    if selected_origin_airport != 'Wszystkie':
        origin_code = [code for code, name in airport_dict.items() if name == selected_origin_airport][0]
        if selected_airlines:
            selected_airline_codes = [code for code, name in airline_dict.items() if name in selected_airlines]
            available_destinations = filtered_flights[
                (filtered_flights['ORIGIN'] == origin_code) &
                (filtered_flights['OP_CARRIER'].isin(selected_airline_codes))
            ]['DEST'].unique().compute()
        else:
            available_destinations = filtered_flights[
                filtered_flights['ORIGIN'] == origin_code
            ]['DEST'].unique().compute()
    else:
        available_destinations = filtered_flights['DEST'].unique().compute()

    available_dest_airports = [airport_dict[code] for code in available_destinations if code in airport_dict]

    selected_dest_airports = st.sidebar.multiselect(
        'Wybierz lotniska docelowe (opcjonalnie)',
        available_dest_airports,
        default=[]
    )

    # Przycisk 'Analizuj'
    analyze_button = st.sidebar.button('Analizuj')

    if analyze_button:
        # Filtrowanie danych na podstawie wybranych kryteriów
        if selected_airlines:
            selected_airline_codes = [code for code, airline in airline_dict.items() if airline in selected_airlines]
            filtered_flights = filtered_flights[filtered_flights['OP_CARRIER'].isin(selected_airline_codes)]

        if selected_origin_airport != 'Wszystkie':
            filtered_flights = filtered_flights[filtered_flights['ORIGIN'] == origin_code]

        if selected_dest_airports:
            dest_codes = [code for code, name in airport_dict.items() if name in selected_dest_airports]
            filtered_flights = filtered_flights[filtered_flights['DEST'].isin(dest_codes)]

        
        filtered_flights = filtered_flights.compute()

        filtered_flights['ARR_DELAY'] = filtered_flights['ARR_DELAY'].fillna(0)

        # Dodanie kolumny 'month_year' do analizy chronologicznej
        filtered_flights['month_year'] = (
            filtered_flights['year'].astype(str) + '-' + filtered_flights['month'].astype(int).astype(str).str.zfill(2)
        )
        filtered_flights = filtered_flights.sort_values(by='month_year')

        # Filtrowanie opóźnionych lotów zgodnie z wybranym progiem opóźnienia
        delayed_flights = filtered_flights[filtered_flights['ARR_DELAY'] > delay_threshold]

        # Filtrowanie lotów opóźnionych z powodu pogody
        weather_delay_flights = delayed_flights[delayed_flights['WEATHER_DELAY'] > 0]

        # Wczytanie współrzędnych geograficznych
        airport_coords = airports[['IATA_CODE', 'LATITUDE', 'LONGITUDE']].compute()

        st.subheader("Liczba lotów wybranych do analizy: {}".format(len(filtered_flights)))
        st.caption(f"Za lot opóźniony uznano lot, który dotarł na lotnisko docelowe z opóźnieniem większym niż {delay_threshold} minut.")

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Wykres 1: Procentowy udział lotów opóźnionych
        st.subheader('Procentowy udział lotów opóźnionych')
        st.caption("Wykres przedstawia procentowy udział lotów opóźnionych w stosunku do wszystkich lotów w analizowanym okresie.")
        def plot_delay_distribution(flights_df):
            delays = flights_df['ARR_DELAY'] > delay_threshold
            delay_counts = delays.value_counts()

            if delay_counts.size == 0:
                st.write("Brak danych do wyświetlenia dla wykresu.")
                return

            labels = ['Na czas', 'Opóźnione']
            colors = ['#003f5c', '#ffa600']

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                delay_counts,
                labels=labels,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                wedgeprops=dict(width=0.3)
            )

            for autotext in ax.texts:
                autotext.set_color('#000000')
                autotext.set_fontsize(5)

            # Dostosowanie etykiet legendy
            ax.legend(labels,
                      title="Rodzaj lotu",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=4,
                      title_fontsize='4')

            ax.axis('equal')
            st.pyplot(fig)

        plot_delay_distribution(filtered_flights)

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        import matplotlib.ticker as mticker

        # Wykres 2: Liczba lotów (Opóźnione vs Na Czas) w badanym okresie
        st.subheader('Liczba lotów z podziałem na loty opóźnione i loty na czas')
        st.caption("Wykres pokazuje liczbę lotów opóźnionych i lotów na czas w każdym miesiącu w analizowanym okresie.")

        def plot_stacked_flights_count(flights_df):
            flights_df['month_year'] = flights_df['month_year'].astype(str)
            delayed_flights_counts = delayed_flights.groupby('month_year').size()
            on_time_flights_counts = flights_df[flights_df['ARR_DELAY'] <= delay_threshold].groupby('month_year').size()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(delayed_flights_counts.index, delayed_flights_counts, label='Opóźnione', color='#ffa600')
            ax.bar(on_time_flights_counts.index, on_time_flights_counts, bottom=delayed_flights_counts, label='Na czas', color='#003f5c')

            ax.set_xlabel('Okres')
            ax.set_ylabel('Liczba lotów')

            # Dodanie separatorów tysięcznych na osi y
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

            if len(delayed_flights_counts.index) > 60:
                years = sorted(set(idx.split('-')[0] for idx in delayed_flights_counts.index))
                year_labels = [f"{year}" for year in years]
                ax.set_xticks(np.arange(0, len(delayed_flights_counts.index), 12))
                ax.set_xticklabels(year_labels, rotation=0)
            else:
                month_names = ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"]
                ax.set_xticks(range(len(delayed_flights_counts.index)))
                ax.set_xticklabels([
                    f"{month_names[int(idx.split('-')[1]) - 1]} {idx.split('-')[0]}" for idx in delayed_flights_counts.index
                ], rotation=90)

            ax.legend()
            st.pyplot(fig)

        plot_stacked_flights_count(filtered_flights)

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        # Wykres 3: Procent opóźnionych lotów (miesięcznie) z linią trendu
        st.subheader('Procent opóźnionych lotów')
        st.caption("Wykres przedstawia miesięczny procent lotów opóźnionych, wraz z linią trendu pokazującą ogólną tendencję w analizowanym okresie.")
        def plot_delayed_flights_percentage(flights_df):
            flights_df['month_year'] = flights_df['month_year'].astype(str)

            monthly_totals = flights_df.groupby('month_year').size()
            monthly_delayed = flights_df[flights_df['ARR_DELAY'] > delay_threshold].groupby('month_year').size()

            percentages = (monthly_delayed / monthly_totals) * 100

            if percentages.empty:
                st.write("Brak danych do wyświetlenia dla wykresu procentowego udziału opóźnionych lotów miesięcznie.")
                return

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x=percentages.index, y=percentages.values, ax=ax, label='Procent opóźnionych lotów', marker='o', color='#003f5c')

            # Obliczanie linii trendu przy użyciu linregress
            x_vals = np.arange(len(percentages.index))
            slope, intercept, r_value, p_value, std_err = linregress(x_vals, percentages.values)
            trendline = slope * x_vals + intercept
            ax.plot(percentages.index, trendline, color='#bc5090', label='Linia trendu')

            ax.set_xlabel('Okres')
            ax.set_ylabel('Opóźnione loty (%)')

            if len(percentages.index) > 60:
                years = sorted(set(idx.split('-')[0] for idx in percentages.index))
                year_labels = [f"{year}" for year in years]
                ax.set_xticks(np.arange(0, len(percentages.index), 12))
                ax.set_xticklabels(year_labels, rotation=0)
            else:
                month_names = ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"]
                ax.set_xticks(range(len(percentages.index)))
                ax.set_xticklabels([
                    f"{month_names[int(idx.split('-')[1]) - 1]} {idx.split('-')[0]}" for idx in percentages.index
                ], rotation=90)

            ax.legend()
            st.pyplot(fig)

        plot_delayed_flights_percentage(filtered_flights)


        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        # Wykres 4: Przyczyny opóźnień lotów
        st.subheader('Przyczyny opóźnień lotów')
        st.caption('Procentowy udział każdej przyczyny opóźnienia został obliczony, dzieląc liczbę opóźnień z danej przyczyny przez całkowitą liczbę opóźnionych lotów, a następnie przeliczając na procenty.')

        delay_columns = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

        delay_sums = delayed_flights[delay_columns].sum()
        total_delay = delay_sums.sum()
        delay_percentages = (delay_sums / total_delay) * 100

        label_map = {
            'CARRIER_DELAY': 'Przyczyny operacyjne',
            'WEATHER_DELAY': 'Pogoda',
            'NAS_DELAY': 'Kontrola ruchu lotniczego',
            'SECURITY_DELAY': 'Bezpieczeństwo',
            'LATE_AIRCRAFT_DELAY': 'Opóźniony samolot z poprzedniego lotu'
        }

        color_map = {
            'CARRIER_DELAY': '#003f5c',
            'WEATHER_DELAY': '#58508d',
            'NAS_DELAY': '#bc5090',
            'SECURITY_DELAY': '#ff6361',
            'LATE_AIRCRAFT_DELAY': '#ffa600'
        }

        fig = go.Figure(data=[go.Pie(
            labels=[label_map[key] for key in delay_percentages.index],
            values=delay_percentages.values,
            marker=dict(colors=[color_map[key] for key in delay_percentages.index]),
            hole=0.3,
            hoverinfo='label+percent'
        )])

        fig.update_layout(
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                font=dict(size=12),
                traceorder='normal'
            ),
            margin=dict(t=50, l=25, r=25, b=25),
            plot_bgcolor='#FFFFFF'
        )

        st.plotly_chart(fig)

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Wykres 5: Średnia i mediana opóźnienia według przyczyny
        st.subheader('Średnia i mediana opóźnienia według przyczyny')
        st.caption('Średnia opóźnienia reprezentuje przeciętny czas opóźnienia w przypadku, gdy dana kategoria przyczyniła się do opóźnienia, natomiast mediana pokazuje wartość środkową.')

        delay_causes_mean = {}
        delay_causes_median = {}

        for col in delay_columns:
            category_delayed_flights = delayed_flights[delayed_flights[col] > 0]

            if category_delayed_flights.shape[0] > 0:
                mean_delay = category_delayed_flights['ARR_DELAY'].mean()
                median_delay = category_delayed_flights['ARR_DELAY'].quantile(0.5)
            else:
                mean_delay = 0
                median_delay = 0

            delay_causes_mean[col] = mean_delay
            delay_causes_median[col] = median_delay

        delay_causes_mean_dict = {label_map[key]: value for key, value in delay_causes_mean.items()}
        delay_causes_median_dict = {label_map[key]: value for key, value in delay_causes_median.items()}

        color_map = {
            'Średnie Opóźnienie': '#003f5c',
            'Mediana Opóźnienia': '#ff6361'
        }

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(delay_causes_mean_dict.keys()),
            y=list(delay_causes_mean_dict.values()),
            name='Średnie Opóźnienie',
            marker=dict(color=color_map['Średnie Opóźnienie'])
        ))

        fig.add_trace(go.Scatter(
            x=list(delay_causes_median_dict.keys()),
            y=list(delay_causes_median_dict.values()),
            mode='markers+text',
            name='Mediana Opóźnienia',
            marker=dict(color=color_map['Mediana Opóźnienia'], size=15),
            text=[round(val, 2) for val in delay_causes_median_dict.values()],
            textposition='top center',
            textfont=dict(color='#FFA600')
        ))

        fig.update_layout(
            yaxis_title='Minuty',
            barmode='group',
        )

        st.plotly_chart(fig)

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Wykres 6: KPIs
        st.subheader('KPIs - Średni czas opóźnienia')
        st.caption('Średni czas opóźnienia obliczono jako średnią arytmetyczną wszystkich czasów opóźnień dla opóźnionych lotów oraz dla tych opóźnionych z powodu pogody.')

        # Obliczanie średniego czasu opóźnienia
        avg_delay_time = delayed_flights['ARR_DELAY'].mean()

        # Obliczanie średniego czasu opóźnienia spowodowanego pogodą
        avg_delay_time_weather = weather_delay_flights['WEATHER_DELAY'].mean()

        # Wyświetlanie wskaźników KPI
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="dla lotów opóźnionych (min.)", value=round(float(avg_delay_time), 2))

        with col2:
            st.metric(label="dla opóźnień spowodowanych pogodą (min.)", value=round(float(avg_delay_time_weather), 2))

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        # Wykres 7: Procent opóźnonych lotów z powodu pogody z linią trendu
        st.subheader('Procentowy udział lotów opóźnionych z powodu pogody we wszystkich opóźnieniach')
        st.caption('Udział obliczono jako stosunek liczby lotów opóźnionych z powodu pogody (opóźnienie z powodu pogody > 0 min.) do ogólnej liczby opóźnionych lotów.')
        def plot_weather_delayed_flights_percentage(flights_df):
            flights_df['month_year'] = flights_df['month_year'].astype(str)

            monthly_delayed = flights_df[flights_df['ARR_DELAY'] > delay_threshold].groupby('month_year').size()
            monthly_weather_delayed = flights_df[flights_df['WEATHER_DELAY'] > 0].groupby('month_year').size()

            # Obliczanie procentu opóźnionych lotów z powodu pogody
            weather_percentages = (monthly_weather_delayed / monthly_delayed) * 100

            if weather_percentages.empty:
                st.write("Brak danych do wyświetlenia dla wykresu procentowego udziału opóźnionych lotów z powodu pogody.")
                return

            fig, ax = plt.subplots(figsize=(12, 6))

            # Wykres: Procent opóźnionych lotów z powodu pogody
            sns.lineplot(x=weather_percentages.index, y=weather_percentages.values, ax=ax, label='Procent opóźnionych lotów z powodu pogody', marker='o', color='#ffa600')

            # Obliczanie i dodawanie linii trendu
            x_vals = np.arange(len(weather_percentages))
            y_vals = weather_percentages.values
            slope = (np.mean(x_vals * y_vals) - np.mean(x_vals) * np.mean(y_vals)) / (np.mean(x_vals**2) - np.mean(x_vals)**2)
            intercept = np.mean(y_vals) - slope * np.mean(x_vals)
            trendline = slope * x_vals + intercept
            ax.plot(weather_percentages.index, trendline, color='#ff6361', label='Linia trendu')

            ax.set_xlabel('Okres')
            ax.set_ylabel('Opóźnione loty z powodu pogody (%)')
            ax.legend()

            # Ustawienia osi X
            if len(weather_percentages.index) > 60:
                years = sorted(set(idx.split('-')[0] for idx in weather_percentages.index))
                year_labels = [f"{year}" for year in years]
                ax.set_xticks(np.arange(0, len(weather_percentages.index), 12))
                ax.set_xticklabels(year_labels, rotation=0)
            else:
                month_names = ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"]
                ax.set_xticks(range(len(weather_percentages.index)))
                ax.set_xticklabels([
                    f"{month_names[int(idx.split('-')[1]) - 1]} {idx.split('-')[0]}" for idx in weather_percentages.index
                ], rotation=90)

            st.pyplot(fig)

        # Wywołanie funkcji rysującej wykres
        plot_weather_delayed_flights_percentage(filtered_flights)

        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        # Wykres 8: Średnie opóźnienie według miesięcy
        st.subheader('Średnie opóźnienia ogólne i spowodowane pogodą zgrupowane według miesięcy')

        # Ensure delayed_flights is a Dask DataFrame and not a Pandas DataFrame
        overall_avg_delay = delayed_flights.groupby('month')['ARR_DELAY'].mean()  # This should be a Dask Series if delayed_flights is a Dask DataFrame
        weather_avg_delay = weather_delay_flights.groupby('month')['WEATHER_DELAY'].mean()  # Same here for weather_delay_flights

        # Now convert them to Pandas using compute() if they are Dask Series
        if isinstance(overall_avg_delay, dd.Series):
            overall_avg_delay = overall_avg_delay.compute()

        if isinstance(weather_avg_delay, dd.Series):
            weather_avg_delay = weather_avg_delay.compute()

        # Map month numbers to names, assuming month_names is defined
        month_names = {
            1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień', 5: 'Maj', 6: 'Czerwiec',
            7: 'Lipiec', 8: 'Sierpień', 9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
        }
        overall_avg_delay.index = overall_avg_delay.index.map(month_names)
        weather_avg_delay.index = weather_avg_delay.index.map(month_names)

        # Ensure all months are represented
        all_months = list(month_names.values())
        overall_avg_delay = overall_avg_delay.reindex(all_months).fillna(0)
        weather_avg_delay = weather_avg_delay.reindex(all_months).fillna(0)

        # Sort months
        sorted_months = sorted(all_months, key=lambda x: list(month_names.values()).index(x))
        overall_avg_delay = overall_avg_delay[sorted_months]
        weather_avg_delay = weather_avg_delay[sorted_months]

        # Create plot
        fig = go.Figure()

        # Add line for overall average delay
        fig.add_trace(go.Scatter(
            x=overall_avg_delay.index,
            y=overall_avg_delay.values,
            mode='lines+markers',
            name='Średnie Opóźnienie Ogólne',
            line=dict(color='#003f5c', width=2),
            marker=dict(size=6)
        ))

        # Add line for weather-related average delay
        fig.add_trace(go.Scatter(
            x=weather_avg_delay.index,
            y=weather_avg_delay.values,
            mode='lines+markers',
            name='Średnie Opóźnienie z Powodu Pogody',
            line=dict(color='#ffa600', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            xaxis_title="Miesiąc",
            yaxis_title="Średnie Opóźnienie (minuty)",
            legend_title="Rodzaj Opóźnienia",
            xaxis=dict(tickvals=all_months, ticktext=all_months)
        )

        st.plotly_chart(fig)


        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        # Wykres 9: Opóźnienie z powodu pogody według portów wylotowych
        # Grupowanie według portów wylotowych
        origin_weather_delays = (
            weather_delay_flights
            .groupby('ORIGIN')['WEATHER_DELAY']
            .mean()
            .reset_index()
        )

        # Grupowanie według portów docelowych
        dest_weather_delays = (
            weather_delay_flights
            .groupby('DEST')['WEATHER_DELAY']
            .mean()
            .reset_index()
        )

        # Dodanie współrzędnych geograficznych
        origin_weather_delays = origin_weather_delays.merge(
            airport_coords,
            left_on='ORIGIN',
            right_on='IATA_CODE',
            how='left'
        )

        dest_weather_delays = dest_weather_delays.merge(
            airport_coords,
            left_on='DEST',
            right_on='IATA_CODE',
            how='left'
        )

        # Filtruj wiersze z NaN w szerokości geograficznej lub długości geograficznej
        origin_weather_delays = origin_weather_delays.dropna(subset=['LATITUDE', 'LONGITUDE'])
        dest_weather_delays = dest_weather_delays.dropna(subset=['LATITUDE', 'LONGITUDE'])

        # Tworzenie heatmap
        st.subheader('Średnie opóźnienie z powodu pogody według portów wylotowych')
        st.caption('Średnie opóźnienie dla każdego portu wylotowego obliczono, sumując czas opóźnień spowodowanych pogodą w danym porcie, a następnie dzieląc przez liczbę lotów opóźnionych z tego powodu.')

        fig_origin_circles = px.scatter_mapbox(
            origin_weather_delays,
            lat='LATITUDE',
            lon='LONGITUDE',
            size='WEATHER_DELAY',
            color='WEATHER_DELAY',
            color_continuous_scale='Oranges',
            size_max=15,
            mapbox_style="open-street-map"
        )

        fig_origin_circles.update_layout(
            coloraxis_colorbar_title='Średnie opóźnienie spowodowane pogodą (min.)',
            mapbox=dict(
                zoom=1
            )
        )

        st.plotly_chart(fig_origin_circles)


        # Wykres 10: Opóźnienie z powodu pogody według portów docelowych
        st.subheader('Średnie opóźnienie z powodu pogody według portów docelowych')
        st.caption('Średnie opóźnienie dla każdego portu docelowego obliczono, sumując czas opóźnień spowodowanych pogodą w danym porcie, a następnie dzieląc przez liczbę lotów opóźnionych z tego powodu.')

        fig_dest_circles = px.scatter_mapbox(
            dest_weather_delays,
            lat='LATITUDE',
            lon='LONGITUDE',
            size='WEATHER_DELAY',
            color='WEATHER_DELAY',
            color_continuous_scale='Oranges',
            size_max=15,
            mapbox_style="open-street-map"
        )

        fig_dest_circles.update_layout(
            coloraxis_colorbar_title='Średnie opóźnienie spowodowane pogodą (min.)',
            mapbox=dict(
                zoom=1
            )
        )

        st.plotly_chart(fig_dest_circles)


        # Wykres 11: 5 tras z największym średnim opóźnieniem z powodu pogody
        # Obliczenie liczby tygodni w wybranym okresie
        start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
        end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(1)
        number_of_weeks = (end_date - start_date).days // 7

        # Ustalenie minimalnej liczby lotów na trasie jako 1 lot na tydzień
        minimal_loty = number_of_weeks * 1

        st.subheader('5 tras z największym średnim opóźnieniem z powodu pogody')
        st.caption(f'Dla tras z co najmniej 1 lotem tygodniowo, czyli z minimum {minimal_loty} lotami w wybranym okresie.')

        trasa_opóźnienia = weather_delay_flights.groupby(['ORIGIN', 'DEST']).agg({
            'WEATHER_DELAY': ['sum', 'count']
        })

        trasa_opóźnienia.columns = ['suma_opóźnień', 'liczba_lotów']

        # Filtrowanie tras, które spełniają warunek minimalnej liczby lotów
        trasa_opóźnienia = trasa_opóźnienia[trasa_opóźnienia['liczba_lotów'] >= minimal_loty]

        trasa_opóźnienia['średnie_opóźnienie'] = trasa_opóźnienia['suma_opóźnień'] / trasa_opóźnienia['liczba_lotów']
        trasa_opóźnienia['średnie_opóźnienie'] = trasa_opóźnienia['średnie_opóźnienie'].round(2)

        trasa_opóźnienia = trasa_opóźnienia.reset_index()
        trasa_opóźnienia['trasa'] = trasa_opóźnienia.apply(lambda x: f"{x['ORIGIN']} -> {x['DEST']}", axis=1)

        top_5_trasy = trasa_opóźnienia.nlargest(5, 'średnie_opóźnienie')
        top_5_trasy = top_5_trasy.sort_values(by='średnie_opóźnienie', ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_5_trasy['trasa'],
            x=top_5_trasy['średnie_opóźnienie'],
            orientation='h',
            marker=dict(color='#ffa600'),
            text=top_5_trasy['średnie_opóźnienie'],
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title='Średnie opóźnienie (minuty)',
            yaxis_title='Trasa',
            yaxis=dict(categoryorder='total ascending')
        )

        st.plotly_chart(fig)
