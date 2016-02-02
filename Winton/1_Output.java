import java.io.*;

public class 1_Output {
    public static void main(String [] args) {
		
		// Input file from which median result needs to be read
        String fileName = "result.txt";
		
		// Output CSV file 
		String opFileName = "1-MedianRepl.csv.csv";
		
		String line = null;
        try {
			
			// File writer. Initialised only once
			FileWriter fileWriter = new FileWriter(opFileName);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
			
			// Header info of output file
			bufferedWriter.write("Id,Predicted");
					
			for (int i = 1; i <= 120000; i++) {
				
				// File reader to be read for each iteration as the values are going to be common
				FileReader fileReader = new FileReader(fileName);
				BufferedReader bufferedReader = new BufferedReader(fileReader);
				
				// Some logger statements to know about the progress of processing
				System.out.println("Writing Line: " + i);
				
				while((line = bufferedReader.readLine()) != null) {
					bufferedWriter.write("\n" + i + line);
				}
				
				// Always close reader file
				bufferedReader.close();  
				fileReader.close();
			}
			
            // Always close writer file
			bufferedWriter.close();
			fileWriter.close();
			
        } catch(FileNotFoundException ex) {
            System.out.println("Unable to open file '" + fileName + "'");                
        } catch(IOException ex) {
            System.out.println("Error reading file '" + fileName + "'");                  
        }
    }
}
