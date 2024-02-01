Attribute VB_Name = "Module2"
Sub incels_sample()
'
'incels_sample Macro
    Application.DisplayAlerts = False
    Dim path As String
    'Changer le path au besoin
    path = "C:\Users\p1115145\Documents\SCI6203\corpus\corpus_filtre_sampled\"
    
    Dim c As Integer
    c = 1
    
    Do While c < 60
    Dim fileName As String
    fileName = "incels_" & CStr(c) & ".xlsx"
    Dim fullPath As String
    fullPath = path & fileName
    Workbooks.Open "" & fullPath & ""
    
    Dim rangeData As Integer
    If c = 1 Then
        rangeData = 1 + (c * 360)
    Else
        rangeData = c * 360
    End If
    
    range("J1").Select
    With Selection.Interior
        .Pattern = xlSolid
        .PatternThemeColor = xlThemeColorAccent1
        .ThemeColor = xlThemeColorAccent6
        .TintAndShade = 0
        .PatternTintAndShade = 0
    End With
    With Selection.Font
        .ThemeColor = xlThemeColorDark1
        .TintAndShade = 0
    End With
    Selection.Font.Bold = True
    ActiveCell.FormulaR1C1 = "échantillonnage"
    
    'Créer une nouvelle colonne où appliquer la formule ALEA() pour
    'assigner une valeur aléatoire à chaque ligne
    range("J2").Select
    ActiveCell.FormulaR1C1 = "=IF(NOT(ISBLANK(RC[-2])),RAND(),"""")"
    Selection.AutoFill Destination:=range("J2:J500000")
    range("J2:J500000").Select
    Columns("J:J").Select
    Selection.Copy
    Selection.PasteSpecial Paste:=xlPasteValues, Operation:=xlNone, SkipBlanks _
        :=False, Transpose:=False
    Application.CutCopyMode = False
    Columns("A:J").Select
    range("J1").Activate
    
    'Trier les données
    ActiveWorkbook.Worksheets("data_incels-2015-2019").Sort.SortFields.Clear
    ActiveWorkbook.Worksheets("data_incels-2015-2019").Sort.SortFields.Add2 Key:=range( _
        "J2:J500000"), SortOn:=xlSortOnValues, Order:=xlAscending, DataOption:= _
        xlSortNormal
    With ActiveWorkbook.Worksheets("data_incels-2015-2019").Sort
        .SetRange range("A1:J500000")
        .Header = xlYes
        .MatchCase = False
        .Orientation = xlTopToBottom
        .SortMethod = xlPinYin
        .Apply
    End With
    
    'Sélectionner les 360 valeurs pour notre échantillon (20 000 au total, / 56 fichiers = 358 lignes par fichier)
    ' (pourquoi 56 fichiers : à l'étape de filtrer pour 2015-2019, on se retrouve avec 3 fichiers vides, pour lesquels aucune donnée
    ' ne se trouvait entre 2015 et 2019)
    range("A1:J360").Select
    Selection.Copy
    Sheets.Add After:=ActiveSheet
    Selection.PasteSpecial Paste:=xlPasteColumnWidths, Operation:=xlNone, _
        SkipBlanks:=False, Transpose:=False
    ActiveSheet.Paste
    range("A1").Select
    Sheets("Feuil1").Select
    Sheets("Feuil1").name = "incels_sample"
    Sheets("data_incels-2015-2019").Select
    Application.CutCopyMode = False
    ActiveWindow.SelectedSheets.Delete
    
    'Copier les données dans le classeur combiné
    range("A2:I360").Select
    Selection.Copy

    Windows("corpus_incels_20k.xlsx").Activate
    ActiveSheet.Paste
    Selection.End(xlDown).Select
    range("A" & CStr(rangeData)).Select

    Windows("" & fileName & "").Activate
    range("A1").Select
    ActiveWorkbook.save
    ActiveWindow.Close
    
    'Passer au classeur suivant
    c = c + 1
    Loop
    
    'Une fois les données extraites de chacun des 60 fichiers, enregistrer et ferme le classeur où sont combinées les données
    'Ce classeur constitue donc notre corpus final pour les données "Incels"
    Windows("corpus_incels_20k.xlsx").Activate
    ActiveWorkbook.save
    ActiveWindow.Close
    
End Sub


