import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ARTIFACT_DIR = path.resolve(SCRIPT_DIR, "..");
const RESULTS_DIR = path.join(ARTIFACT_DIR, "results");
const OUTPUT_XLSX = path.join(ARTIFACT_DIR, "long_context_8k_bar_data.xlsx");

const SPEEDUP_CSV = path.join(RESULTS_DIR, "long_context_8k_normalized_speedup.csv");
const ENERGY_CSV = path.join(RESULTS_DIR, "long_context_8k_normalized_energy_efficiency.csv");
const LONG_CSV = path.join(RESULTS_DIR, "long_context_8k_points.csv");

const COLORS = {
  titleFill: "#E6F5F1",
  headerFill: "#1A759F",
  headerFont: "#FFFFFF",
  border: "#D9D9D9",
};


function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const nextChar = text[index + 1];

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        value += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && nextChar === "\n") {
        index += 1;
      }
      row.push(value);
      if (row.length > 1 || row[0] !== "") {
        rows.push(row);
      }
      row = [];
      value = "";
      continue;
    }

    value += char;
  }

  if (value.length > 0 || row.length > 0) {
    row.push(value);
    rows.push(row);
  }

  return rows;
}


function wideRowsFromCsv(rows) {
  const [header, ...body] = rows;
  return [
    header,
    ...body.map((line) => [
      line[0],
      ...line.slice(1).map((entry) => Number.parseFloat(entry)),
    ]),
  ];
}


function longRowsFromCsv(rows) {
  const [header, ...body] = rows;
  return [
    header,
    ...body.map((line) => [
      line[0],
      line[1],
      line[2],
      line[3] === "" ? "" : Number.parseInt(line[3], 10),
      line[4] === "" ? "" : Number.parseInt(line[4], 10),
      Number.parseFloat(line[5]),
    ]),
  ];
}


function columnName(columnIndexZeroBased) {
  let dividend = columnIndexZeroBased + 1;
  let name = "";
  while (dividend > 0) {
    const modulo = (dividend - 1) % 26;
    name = String.fromCharCode(65 + modulo) + name;
    dividend = Math.floor((dividend - modulo) / 26);
  }
  return name;
}


function rangeAddress(startRowOneBased, startColZeroBased, rowCount, colCount) {
  const startCol = columnName(startColZeroBased);
  const endCol = columnName(startColZeroBased + colCount - 1);
  const endRow = startRowOneBased + rowCount - 1;
  return `${startCol}${startRowOneBased}:${endCol}${endRow}`;
}


function writeTitle(sheet, title, columnCount) {
  const titleRangeAddress = rangeAddress(1, 0, 1, columnCount);
  sheet.mergeCells(titleRangeAddress);
  const titleRange = sheet.getRange(titleRangeAddress);
  titleRange.values = [[title]];
  titleRange.format = {
    fill: COLORS.titleFill,
    font: { bold: true, size: 16, color: "#111827" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  titleRange.format.rowHeightPx = 30;
}


function writeWideSheet(workbook, sheetName, title, rows, tableName) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const columnCount = rows[0].length;
  writeTitle(sheet, title, columnCount);

  const dataAddress = rangeAddress(3, 0, rows.length, columnCount);
  const dataRange = sheet.getRange(dataAddress);
  dataRange.values = rows;
  dataRange.format.borders = { preset: "all", style: "thin", color: COLORS.border };

  const headerRange = sheet.getRange(rangeAddress(3, 0, 1, columnCount));
  headerRange.format = {
    fill: COLORS.headerFill,
    font: { bold: true, color: COLORS.headerFont },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };

  const numericAddress = rangeAddress(4, 1, rows.length - 1, columnCount - 1);
  sheet.getRange(numericAddress).format.numberFormat = "0.000000";
  sheet.getRange(numericAddress).format.horizontalAlignment = "center";
  sheet.getRange(rangeAddress(4, 0, rows.length - 1, 1)).format.font = { bold: true };

  for (let column = 0; column < columnCount; column += 1) {
    const widthPx = column === 0 ? 170 : 95;
    sheet.getRange(`${columnName(column)}:${columnName(column)}`).format.columnWidthPx = widthPx;
  }
  sheet.freezePanes.freezeRows(3);
  sheet.tables.add(dataAddress, true, tableName);
  return sheet;
}


function writeLongSheet(workbook, rows) {
  const sheet = workbook.worksheets.add("All Bars Long");
  sheet.showGridLines = false;
  const columnCount = rows[0].length;
  const dataAddress = rangeAddress(1, 0, rows.length, columnCount);
  const dataRange = sheet.getRange(dataAddress);
  dataRange.values = rows;
  dataRange.format.borders = { preset: "all", style: "thin", color: COLORS.border };

  sheet.getRange(rangeAddress(1, 0, 1, columnCount)).format = {
    fill: COLORS.headerFill,
    font: { bold: true, color: COLORS.headerFont },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(rangeAddress(2, 3, rows.length - 1, 2)).format.numberFormat = "0";
  sheet.getRange(rangeAddress(2, 5, rows.length - 1, 1)).format.numberFormat = "0.000000";

  const widths = [190, 170, 110, 125, 125, 95];
  for (let column = 0; column < columnCount; column += 1) {
    sheet.getRange(`${columnName(column)}:${columnName(column)}`).format.columnWidthPx = widths[column];
  }
  sheet.freezePanes.freezeRows(1);
  sheet.tables.add(dataAddress, true, "AllBarsLongTable");
  return sheet;
}


async function main() {
  const [speedupText, energyText, longText] = await Promise.all([
    fs.readFile(SPEEDUP_CSV, "utf8"),
    fs.readFile(ENERGY_CSV, "utf8"),
    fs.readFile(LONG_CSV, "utf8"),
  ]);

  const speedupRows = wideRowsFromCsv(parseCsv(speedupText));
  const energyRows = wideRowsFromCsv(parseCsv(energyText));
  const longRows = longRowsFromCsv(parseCsv(longText));

  const workbook = Workbook.create();

  writeWideSheet(
    workbook,
    "Speedup Bars",
    "Long-context 8K normalized speedup: each plotted bar value",
    speedupRows,
    "SpeedupBarsTable",
  );
  writeWideSheet(
    workbook,
    "Energy Bars",
    "Long-context 8K normalized energy efficiency: each plotted bar value",
    energyRows,
    "EnergyBarsTable",
  );
  writeLongSheet(workbook, longRows);

  const check = await workbook.inspect({
    kind: "table",
    range: "'Speedup Bars'!A3:K9",
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 12,
  });
  console.log(check.ndjson);

  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });
  console.log(errors.ndjson);

  for (const sheetName of ["Speedup Bars", "Energy Bars", "All Bars Long"]) {
    const preview = await workbook.render({
      sheetName,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    console.log(`Rendered ${sheetName}: ${new Uint8Array(await preview.arrayBuffer()).length} bytes`);
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(OUTPUT_XLSX);

  const exported = await FileBlob.load(OUTPUT_XLSX);
  const reopenedWorkbook = await SpreadsheetFile.importXlsx(exported);
  const reopenedCheck = await reopenedWorkbook.inspect({
    kind: "table",
    range: "'Energy Bars'!A3:K9",
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 12,
  });
  console.log(reopenedCheck.ndjson);
  console.log(`Saved ${OUTPUT_XLSX}`);
}


await main();
