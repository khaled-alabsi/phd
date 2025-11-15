 i dont like the struckture of the note book, i want to have main sections and sub section, one for import and data prepration and one for code that have custom functioin section and plots function, ml methode section and so on.. then a anomaly detectioin section that only have function call and some print lines, then evaluation section and then results interprtation section.

 test every section/sub section for error for each refactoring step before continuing refactoring.


---

as you can see in code i load the Rdata then i reduce it, which is not efficient, i want to be able to choose in the global configuration whether to load the full dataset or a reduced version, the reduced version, can be exported when i use the Rdata  then reduce it. so the code should be still flexible enough to handle both cases without major changes. maybe you should move the global configuration section to the top before the imports. The exported data should be in CSV format.