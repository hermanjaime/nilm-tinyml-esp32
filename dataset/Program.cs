
using System;
using System.IO;
using System.IO.Ports;
using System.Linq;

namespace Dataset_Collector_NILM
{
    internal class Program
    {
        static void Main()
        {
            string[] featureList = new string[]
            {
                "None",
                "Fan",
                "Blender",
                "Hair Dryer",
                "Fan, Blender",
                "Fan, Hair Dryer",
                "Blender, Hair Dryer",
                "Fan, Blender, Hair Dryer",
            };

            bool loop = true;

            while (loop)
            {
                Console.WriteLine("Select the load combination for data collection:");

                for (int i = 0; i < featureList.Length; i++)
                {
                    Console.WriteLine($"{i + 1}. {featureList[i]}");
                }

                Console.Write("Option (1-8): ");

                if (!int.TryParse(Console.ReadLine(), out int choice) ||
                    choice < 1 || choice > featureList.Length)
                {
                    Console.WriteLine("Invalid option.");
                    return;
                }

                string selectedFeature = featureList[choice - 1].ToLower();

                string[] activeFeatures = new string[]
                {
                    "fan",
                    "blender",
                    "hair dryer"
                };

                bool[] featureState = activeFeatures
                    .Select(f => selectedFeature.Contains(f))
                    .ToArray();

                string portName = "COM6"; // change according to your serial port
                string filename = "Dataset_NILM.csv";

                try
                {
                    if (!File.Exists(filename))
                    {
                        using (var writer = new StreamWriter(filename, append: false))
                        {
                            writer.WriteLine("adc,fan,blender,hair dryer");
                        }
                    }

                    int nSamplesRecorded = 0;

                    using (SerialPort serialPort = new SerialPort(portName, 500000))
                    {
                        serialPort.ReadTimeout = 3000;
                        serialPort.Open();

                        Console.WriteLine($"Reading data from port {portName}");
                        Console.WriteLine($"Selected load: {selectedFeature}");
                        Console.WriteLine($"Saving to: {filename}");

                        using (StreamWriter writer = new StreamWriter(filename, append: true))
                        {
                            while (nSamplesRecorded < 100000)
                            {
                                try
                                {
                                    string line = serialPort.ReadLine().Trim();

                                    if (!string.IsNullOrWhiteSpace(line))
                                    {
                                        if (int.TryParse(line, out int adc))
                                        {
                                            if (adc >= 0 && adc <= 4095)
                                            {
                                                string row =
                                                    $"{adc}," +
                                                    $"{(featureState[0] ? 1 : 0)}," +
                                                    $"{(featureState[1] ? 1 : 0)}," +
                                                    $"{(featureState[2] ? 1 : 0)}";

                                                writer.WriteLine(row);
                                                Console.WriteLine(row);

                                                nSamplesRecorded++;
                                            }
                                        }
                                    }
                                }
                                catch (TimeoutException)
                                {
                                    Console.WriteLine("Timeout while reading from the COM port.");
                                    break;
                                }
                            }

                            Console.WriteLine("Data collection completed.");
                        }
                    }
                }
                catch (UnauthorizedAccessException)
                {
                    Console.WriteLine("Error: COM port is in use. Close the Serial Monitor.");
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"I/O error: {ex.Message}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"General error: {ex.Message}");
                }

                Console.Write("Do you want to collect another load? (y/n): ");
                string resposta = Console.ReadLine();

                loop = resposta.Trim().ToUpper() == "Y";
            }
        }
    }
}