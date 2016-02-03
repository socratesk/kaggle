import java.io.*;
import java.util.*;

public class 3_Csvfile {

public static void main(String[] args) {
    Csvfile obj = new Csvfile();
    obj.run();
}

private String getAdjustedRow(String row) {
	if (row.length() == 1)
		return "00000" + row;
	if (row.length() == 2)
		return "0000" + row;
	if (row.length() == 3)
		return "000" + row;
	if (row.length() == 4)
		return "00" + row;
	if (row.length() == 5)
		return "0" + row;
	return row;
}

public void run() {
	
	// Output filename
	String opFileName = "3-MovingAverage.csv";
	
	// input filename
	String csvFile = "result.csv";
	
  BufferedReader br = null;
  String line = "";
  String cvsSplitBy = ",";

  try {
		
	  // Initialise File Writer
	  FileWriter fileWriter = new FileWriter(opFileName);
    BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
		
		// Prepare column header
		bufferedWriter.write("Id,Predicted");
		
		// Create empty object for for sorted mapping
    Map<String, String> maps = new TreeMap<String, String>();
		
		// Initialie File Reader and read csv file
    br = new BufferedReader(new FileReader(csvFile));
		
		// Loop through the CSV file and populate sorted map.
		// The final map will have all the keys in natural sort order
    while ((line = br.readLine()) != null) {
      String[] values = line.split(cvsSplitBy);
			
			String masterKey = values[0]; // Ex: 1_01, 200_32
			String[] keysplits = masterKey.split("_");
			String row = keysplits[0];  // Ex: 1, 10, 200
			String adjustedRow = getAdjustedRow(row);
			String correctedKey = adjustedRow + "_" + keysplits[1];
			
      maps.put(correctedKey, values[1]);
    }
		System.out.println("Writing to SortedMap completed and writing to output file started...");
		
		// Extract values from Sorted map, loop through it, and wrtie to final output file
		int i = 0;
		SortedSet<String> keys = new TreeSet<String>(maps.keySet());
		for (String key : keys) {
			
			// Some log statements to monitor the process.
			i++;
			if ((i % 1000) == 0) {
				System.out.println("Processed.... " + i);
			}
			
			String value = maps.get(key);
			if (key.contains("_0"))
				key = key.replace("_0", "_"); // Converts 1_01 to 1_1
			
			String[] keysplits = key.split("_");
			int row = Integer.parseInt(keysplits[0]);  // Ex: Converts "000001" to 000001
			String correctedKey = "" + row + "_" + keysplits[1];
			
			// Write to output CSV file
			bufferedWriter.write("\n" + correctedKey + "," + value.trim());
		}
		  // Always close file handlers
			bufferedWriter.close();
			fileWriter.close();
			
    } catch (FileNotFoundException e) {
        e.printStackTrace();
    } catch (IOException e) {
        e.printStackTrace();
    } finally {
        if (br != null) {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    System.out.println("Done");
}
}
