using System;

namespace Chapter15
{
    public class NullableReferenceTypes
    {
        // an alternate method for enabling nullable reference types
#nullable enable

        public string? Nullable { get; set; }
        // Will display warning CS8618 because it is not initialized
        public string NotNullable { get; set; }

        public void TestNullability()
        {
            // Should show warning CS8602
            Console.WriteLine(Nullable.Length);

            // Should not show any error
            Console.WriteLine(NotNullable.Length);

            if (Nullable != null)
            {
                // Because null was checked
                // should be perfectly valid
                Console.WriteLine(Nullable.Length);
            }

            // Bang operator (!) lets compiler know that you 
            // know what's best and it should keep quiet
            Console.WriteLine(Nullable!.Length);
        }
    }
}
