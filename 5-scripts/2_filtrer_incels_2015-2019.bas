Attribute VB_Name = "Module1"
Sub filtrer_incels_2015_2019_nettoyer()
'
'filtrer_incels Macro

'Cette macro-commande VBA lit en boucle chacun des 60 fichiers csv contenant notre corpus et y applique un filtre pour ne retenir que les
'données provenant des subreddits catégorisés comme étant 'Incels' ainsi que les posts publiés entre 2015-2019

'Le filtre appliqué permet également d'exclure les posts supprimés (dont le contenu = "[Removed]" ou "[Deleted]"

'(voir cette source pour la solution permettant de creer une variable où stocker le nom du fichier et lire les fichiers csv en boucle:
'https://www.mrexcel.com/board/threads/vba-create-power-query-with-source-and-name-as-variable-from-excel-sheet.1131242/ )
    
    Dim c As Integer
    c = 2

    Do While c < 60
    
    Workbooks.Add
    
    'Ouvrir le fichier .csv et y lire les données
    Dim requete As String
    requete = "reddit_500k_" & CStr(c)
    Source = "C:\Users\p1115145\Documents\SCI6203\corpus\csv\" & requete & ".csv"
    
    ActiveWorkbook.Queries.Add name:=requete, Formula:= _
        "let" & Chr(13) & "" & Chr(10) & "    Source = Csv.Document(File.Contents(""" & Source & """),[Delimiter="","", Columns=7, Encoding=65001, QuoteStyle=QuoteStyle.Csv])," & Chr(13) & "" & Chr(10) & "    #""En-têtes promus"" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])," & Chr(13) & "" & Chr(10) & "    #""Type modifié"" = Table.TransformColumnTypes(#""En-têtes promus"",{{""author"", type te" & _
        "xt}, {""date_post"", Int64.Type}, {""id_post"", type text}, {""number_post"", type text}, {""subreddit"", type text}, {""text_post"", type text}, {""thread"", type text}})" & Chr(13) & "" & Chr(10) & "in" & Chr(13) & "" & Chr(10) & "    #""Type modifié"""
    ActiveWorkbook.Worksheets.Add
    
    With ActiveSheet.ListObjects.Add(SourceType:=0, Source:= _
        "OLEDB;Provider=Microsoft.Mashup.OleDb.1;Data Source=$Workbook$;Location=""" & requete & """;Extended Properties=""""" _
        , Destination:=range("$A$1")).QueryTable
        .CommandType = xlCmdSql
        .CommandText = Array("SELECT * FROM [" & requete & "]")
        .RowNumbers = False
        .FillAdjacentFormulas = False
        .PreserveFormatting = True
        .RefreshOnFileOpen = False
        .BackgroundQuery = True
        .RefreshStyle = xlInsertDeleteCells
        .SavePassword = False
        .SaveData = True
        .AdjustColumnWidth = True
        .RefreshPeriod = 0
        .PreserveColumnInfo = True
        .ListObject.DisplayName = requete
        .Refresh BackgroundQuery:=False
    End With
    
    Application.CommandBars("Queries and Connections").Visible = False
    Application.DisplayAlerts = False
    Columns("A:G").Select
    range("H1").Select
    
    '-------------------------
    
    'Créer deux nouvelles feuilles, (1) pour aller spécifier notre critère d'extraction, (2) pour déposer les données répondant au critère
    range("E1").Select
    Selection.Copy
    Sheets("Feuil1").Select
    ActiveSheet.Paste
    
    Sheets("Feuil1").Select
    Sheets("Feuil1").name = "crit"
    
    Sheets("Feuil2").Select
    Sheets("Feuil2").name = "data"
    
    'Convertir les dates UNIX time stamp en format AAAA-MM-JJ
    Sheets("data").Select
    Columns("C:C").Select
    Selection.Insert Shift:=xlToRight, CopyOrigin:=xlFormatFromLeftOrAbove
    range("C1").Select
    ActiveCell.FormulaR1C1 = "date"
    Columns("D:D").Select
    Selection.Insert Shift:=xlToRight, CopyOrigin:=xlFormatFromLeftOrAbove
    range("D1").Select
    ActiveCell.FormulaR1C1 = "année"
    
    range("C2").Select
    ActiveCell.FormulaR1C1 = "=(RC[-1]/86400)+DATE(1970,1,1)"
    Columns("C:C").Select
    Selection.NumberFormat = "m/d/yyyy"
    range("D2").Select
    ActiveCell.FormulaR1C1 = "=YEAR([@date])"
    
    range("A1:I1").Select
    Selection.Copy
    Sheets.Add After:=ActiveSheet
    Sheets("Feuil3").Select
    range("A1").Select
    Selection.PasteSpecial Paste:=xlPasteColumnWidths, Operation:=xlNone, _
        SkipBlanks:=False, Transpose:=False
    ActiveSheet.Paste
    
    Sheets("Feuil3").Select
    Sheets("Feuil3").name = "data_incels-2015-2019"
    
    
    'Spécifier le critère d'extraction :
    ' -les 22 subreddits étiquettés "Incel"
    ' -publiés entre 2015-2019
    ' -exclure les publications qui ont été supprimées (textpost = "[Removed]" ou "[Deleted]")
    
    '--- Subreddits 'Incels'
    Sheets("crit").Select
    range("A2").Select
    ActiveCell.FormulaR1C1 = "braincels"
    range("A3").Select
    ActiveCell.FormulaR1C1 = "askanincel"
    range("A4").Select
    ActiveCell.FormulaR1C1 = "blackpillscience"
    range("A5").Select
    ActiveCell.FormulaR1C1 = "incelswithouthate"
    range("A6").Select
    ActiveCell.FormulaR1C1 = "foreveralone"
    range("A7").Select
    ActiveCell.FormulaR1C1 = "malecels"
    range("A8").Select
    ActiveCell.FormulaR1C1 = "maleforeveralone"
    range("A9").Select
    ActiveCell.FormulaR1C1 = "1ncels"
    range("A10").Select
    ActiveCell.FormulaR1C1 = "incelspurgatory"
    range("A11").Select
    ActiveCell.FormulaR1C1 = "truecels"
    range("A12").Select
    ActiveCell.FormulaR1C1 = "incelbrotherhood"
    range("A13").Select
    ActiveCell.FormulaR1C1 = "incelspurgatory"
    range("A14").Select
    ActiveCell.FormulaR1C1 = "lonelynonviolentmen"
    range("A15").Select
    ActiveCell.FormulaR1C1 = "foreveraloneteens"
    range("A16").Select
    ActiveCell.FormulaR1C1 = "foreveralonelondon"
    range("A17").Select
    ActiveCell.FormulaR1C1 = "gaycel"
    range("A18").Select
    ActiveCell.FormulaR1C1 = "incelselfies"
    range("A19").Select
    ActiveCell.FormulaR1C1 = "gymcels"
    range("A20").Select
    ActiveCell.FormulaR1C1 = "foreverunwanted"
    range("A21").Select
    ActiveCell.FormulaR1C1 = "inceldense"
    range("A22").Select
    ActiveCell.FormulaR1C1 = "supportcel"
    range("A23").Select
    ActiveCell.FormulaR1C1 = "foreveralonedating"
    
    Columns("A:A").EntireColumn.AutoFit
    
    '--- Années 2015-2019
    Sheets("data").Select
    range("D1").Select
    Selection.Copy
    Sheets("crit").Select
    range("B1").Select
    ActiveSheet.Paste
    range("C1").Select
    ActiveSheet.Paste
    range("B2").Select
    ActiveCell.FormulaR1C1 = ">=2015"
    range("C2").Select
    ActiveCell.FormulaR1C1 = "<=2019"
    range("B2").Select
    Selection.Copy
    range("B3").Select
    ActiveSheet.Paste
    range("B2:B3").Select
    Application.CutCopyMode = False
    Selection.AutoFill Destination:=range("B2:B23"), Type:=xlFillDefault

    range("C2").Select
    Selection.Copy
    range("C3").Select
    ActiveSheet.Paste
    range("C2:C3").Select
    Application.CutCopyMode = False
    Selection.AutoFill Destination:=range("C2:C23"), Type:=xlFillDefault

    '--- Exclure [Removed] et [Deleted]
    Sheets("data").Select
    range("H1").Select
    Selection.Copy
    Sheets("crit").Select
    range("D1").Select
    ActiveSheet.Paste
    range("E1").Select
    ActiveSheet.Paste
    range("D2").Select
    ActiveCell.FormulaR1C1 = "<>[removed]"
    Columns("D:D").EntireColumn.AutoFit
    range("E2").Select
    ActiveCell.FormulaR1C1 = "<>[deleted]"
    range("D2:E2").Select
    Columns("E:E").EntireColumn.AutoFit
    Selection.AutoFill Destination:=range("D2:E23"), Type:=xlFillDefault
    range("D2:E23").Select

    '---------------------
    
    'Filtrer les données
    Sheets("data_incels-2015-2019").Select
    Sheets("data").Columns("A:I").AdvancedFilter Action:=xlFilterCopy, _
        CriteriaRange:=Sheets("crit").range("A1:E23"), CopyToRange:=range("A1:I1") _
        , Unique:=False
    
    'Supprimer les feuilles qui ne nous serviront plus (données non filtrées et critères d'extaction)
    Sheets("data").Select
    ActiveWindow.SelectedSheets.Delete
    Sheets("crit").Select
    ActiveWindow.SelectedSheets.Delete
    
    'Enregistrer le classeur
    range("A1").Select
    Dim save As String
    save = "incels_" & CStr(c) & ".xlsx"
    ActiveWorkbook.SaveAs fileName:= _
        "C:\Users\p1115145\Documents\SCI6203\corpus\corpus_filtre\" & save, FileFormat _
        :=xlOpenXMLWorkbook, CreateBackup:=False
    
    ActiveWindow.Close

    'Passer au classeur suivant
    c = c + 1
    Loop
    
    
End Sub

