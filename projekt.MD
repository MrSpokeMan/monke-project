# Projekt Generative AI z Wykorzystaniem LLM i Baz Wektorowych

## Cel Projektu

Stworzenie aplikacji, która:

- **Zbiera dane tekstowe** z wybranych źródeł internetowych.
- **Czyści dane** z reklam i innych głupot.
- **Przetwarza i indeksuje dane** za pomocą modeli językowych (LLM) i przechowuje reprezentacje wektorowe w bazie wektorowej.
- **Umożliwia użytkownikom zadawanie pytań** i otrzymywanie odpowiedzi na podstawie zebranych danych.

## Moduły Projektu

1. **Input - Web Scraper/Image Processing/Inne zrodlo**

   - Zbieranie danych z wybranych stron internetowych.
   - Wykorzystanie bibliotek takich jak `BeautifulSoup`, ale chętnie zobacze gotowe scrapery? Może takie które już przygotowują dane pod LLM?
   - Oczyszczanie i przygotowanie danych do dalszego przetwarzania.
2. **Przetwarzanie Tekstu i Generowanie Embeddingów**

   - Wykorzystanie modeli LLM do tworzenia reprezentacji wektorowych tekstu.
   - Implementacja pipeline'u przetwarzania danych.
3. **Baza Wektorowa**

   - Przechowywanie embeddingów w bazie wektorowej (np. Pinecone, Milvus)
4. **Interfejs Użytkownika**

   - Prosty interfejs (CLI lub webowy) umożliwiający komunikację z użytkownikiem.
   - Użycie FastAPI
5. **Moduł Odpowiadania na Pytania**

   - Wykorzystanie LLM do generowania odpowiedzi na podstawie jakiegoś źródła.
   - Integracja wyników wyszukiwania z bazy wektorowej z generowanymi odpowiedziami.
   - Cel optymalizacji - jak najmniejsza halucynacja LLM.

## Etapy Realizacji

1. **Planowanie Projektu**

   - Omówienie celów i zakresu projektu.
   - **Miejsce na Wasze pomysły:** Zachęta do przedstawienia własnych koncepcji i rozszerzeń projektu.
2. **Implementacja Modułów**

   - Prace nad poszczególnymi modułami zgodnie z podziałem zadań.
   - Regularne konsultacje - Discord.
   - **Miejsce na Wasze pomysły:** Możliwość wyboru narzędzi i rozwiązań technologicznych ułatwiających te prace.
3. **Integracja Systemu**

   - Połączenie wszystkich modułów w spójną aplikację.
   - Testowanie działania całości i eliminacja błędów.
   - **Miejsce na Wasze pomysły:** Sugestie dotyczące usprawnienia komunikacji między modułami.
4. **Testowanie i Optymalizacja**

   - Przeprowadzenie testów funkcjonalnych i wydajnościowych.
   - Optymalizacja kodu i zasobów.
   - **Miejsce na Wasze pomysły:** Propozycje dodatkowych funkcji lub usprawnień zwiększających wartość aplikacji.
5. **Prezentacja i Podsumowanie**

   - Przygotowanie prezentacji wyników prac zespołu.
   - Omówienie napotkanych wyzwań i sposobów ich rozwiązania.
   - **Miejsce na Wasze pomysły:** Dyskusja nad możliwością kontynuacji projektu lub jego komercjalizacji.

## Technologie i Narzędzia

- **Język programowania:** Python
- **Biblioteki i Frameworki:**
  - Web Scraping: `BeautifulSoup`,
  - Przetwarzanie języka: `Transformers`, `OpenAI API`, `Any Other LLM API`
  - Bazy wektorowe: `Pinecone`, `Milvus`
  - Framework webowy: `FastAPI`, `Django (ewentualnie)`
- **Kontrola wersji:** Git (GitHub lub GitLab), lepiej GH

## Dodatkowe Uwagi/Zrodla

- https://github.com/langchain-ai/langchain
- https://www.langchain.com/langgraph
- Uzupelnic
