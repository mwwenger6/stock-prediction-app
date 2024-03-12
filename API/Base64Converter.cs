using System;
using System.Text;

namespace Stock_Prediction_API
{
    public static class Base64Converter
    {
        public static string ToBase64(string plaintext) =>
            Convert.ToBase64String(Encoding.UTF8.GetBytes(plaintext));

        public static string FromBase64(string encodedText) =>
            Encoding.UTF8.GetString(Convert.FromBase64String(encodedText));
    }
}
