using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Extensions
{
    public static partial class Extensions
    {
        /// <summary>
        /// Converts the elements of an array into a well-formatted list.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="a">The array to format into a list.</param>
        /// <param name="separator">The separator to use to separate items in the list.</param>
        /// <param name="elementToString">The method for converting individual array elements to strings.</param>
        /// <returns>A list of the elements in the array as a string.</returns>
        public static string ToStringList<T>(this IList<T> a, string separator, Func<T, string> elementToString)
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < a.Count; ++i)
                builder.Append(elementToString(a[i]) + separator);

            int sepLen = separator.Length;
            if (builder.Length >= sepLen)
                builder.Remove(builder.Length - sepLen, sepLen);

            return builder.ToString();
        }
        /// <summary>
        /// Converts the elements of an array into a well-formatted list.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="a">The array to format into a list.</param>
        /// <param name="separator">The separator to use to separate items in the list.</param>
        /// <param name="lastSeparator">The separator to use in the list between the last two elements.</param>
        /// <param name="elementToString">The method for converting individual array elements to strings.</param>
        /// <returns>A list of the elements in the array as a string.</returns>
        public static string ToStringList<T>(this IList<T> a, string separator, string lastSeparator, Func<T, string> elementToString)
        {
            int countMin2 = a.Count - 2;

            StringBuilder builder = new StringBuilder();
            string sep = separator;
            for (int i = 0; i < countMin2; ++i)
                builder.Append(elementToString(a[i]) + sep);
            
            if (countMin2 >= 0)
                builder.Append(elementToString(a[countMin2]) + lastSeparator);

            ++countMin2;
            if (countMin2 >= 0)
                builder.Append(elementToString(a[countMin2]));

            return builder.ToString();
        }
        /// <summary>
        /// Converts the elements of an array into a well-formatted list.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="a">The array to format into a list.</param>
        /// <param name="separator">The separator to use to separate items in the list.</param>
        /// <param name="elementToString">The method for converting individual array elements to strings.</param>
        /// <returns>A list of the elements in the array as a string.</returns>
        public static string ToStringList<T>(this IList<T> a, string separator, Func<T, int, string> elementToString)
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < a.Count; ++i)
                builder.Append(elementToString(a[i], i) + separator);

            int sepLen = separator.Length;
            if (builder.Length >= sepLen)
                builder.Remove(builder.Length - sepLen, sepLen);

            return builder.ToString();
        }
        /// <summary>
        /// Converts the elements of an array into a well-formatted list.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="a">The array to format into a list.</param>
        /// <param name="separator">The separator to use to separate items in the list.</param>
        /// <param name="elementToString">The method for converting individual array elements to strings.</param>
        /// <returns>A list of the elements in the array as a string.</returns>
        public static string ToStringList<T>(this IList<T> a, string separator, string lastSeparator, Func<T, int, string> elementToString)
        {
            int countMin2 = a.Count - 2;

            StringBuilder builder = new StringBuilder();
            string sep = separator;
            for (int i = 0; i < a.Count; ++i)
                builder.Append(elementToString(a[i], i) + sep);

            if (countMin2 >= 0)
                builder.Append(elementToString(a[countMin2], countMin2) + lastSeparator);

            ++countMin2;
            if (countMin2 >= 0)
                builder.Append(elementToString(a[countMin2], countMin2));

            return builder.ToString();
        }
        /// <summary>
        /// Converts the elements of an array into a well-formatted list.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="a">The array to format into a list.</param>
        /// <param name="separator">The separator to use to separate items in the list.</param>
        /// <returns>A list of the elements in the array as a string.</returns>
        public static string ToStringList<T>(this T[] a, string separator)
            => ToStringList(a, separator, o => o.ToString());
        /// <summary>
        /// Converts the elements of an array into a well-formatted list.
        /// </summary>
        /// <typeparam name="T">The type of elements in the array.</typeparam>
        /// <param name="a">The array to format into a list.</param>
        /// <param name="separator">The separator to use to separate items in the list.</param>
        /// <param name="lastSeparator">The separator to use in the list between the last two elements.</param>
        /// <returns>A list of the elements in the array as a string.</returns>
        public static string ToStringList<T>(this T[] a, string separator, string lastSeparator)
            => ToStringList(a, separator, lastSeparator, o => o.ToString());
    }
}
