-----------------------------------------------------------
--------------------   DATEIEN:  --------------------------
-----------------------------------------------------------

Die Datei "pa1.py" enthält den wichtigsten (mathematischen) Teil von unserer
Lösung.

Die anderen Dateien nutzen diese Funktionen, um Bilder zu erzeugen.

-----------------------------------------------------------
--------------------   NUTZUNG:  --------------------------
-----------------------------------------------------------

Um den Laplace-Operator für n=5, m=7 zu visualisieren, nutzen Sie:

python visualizeLaplace.py

-----------------------------------------------------------

Um alle Bilder (Wasser und Himmel) anzuzeigen, nutzen Sie entweder:

python run_show_images.py

oder:

python run_show_images.py <thread_cnt>

zum Beispiel:

python run_show_images.py 2

Per Default nutzt das Programm bis zu 6 Threads. Das ist schnell, aber braucht
zu viel RAM (~7GB) auf einmal. Wenn sie weniger RAM haben, beschränken sie
die Anzahl. (Wir prüfen die Anzahl von CPU-Kernen nicht, dazu brauchen wir ein
anderes Paket. Wenn ihr Prozessor weniger Kerne als 6 hat, können sie die
Anzahl manuell beschränken.)

-----------------------------------------------------------

Falls es ein Problem mit der Anzeige von matplotlib-Figuren gibt, können Sie die
Bilder mit dem folgenden Befehl direkt als Dateien speichern:

python run_save_images.py

oder:

python run_save_images.py <thread_cnt>

Die Dateinamen (oder Präfixe) können in run_save_images.py bearbeitet werden.

-----------------------------------------------------------

Wir laden auch unsere Tests in "test_pa1.py" hoch, sie werden wie folgt
durchgeführt: 

python -m pytest --verbose

Dazu brauchen sie das pytest-Modul.

-----------------------------------------------------------
--------------------   GRUPPE:   --------------------------
-----------------------------------------------------------

Cem Gündoğdu
Gaeun Kim
Zhanel Sainova

-----------------------------------------------------------
