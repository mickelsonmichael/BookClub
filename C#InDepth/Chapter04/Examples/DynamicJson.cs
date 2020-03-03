using System;
using System.Dynamic;
using System.Net;
using System.Net.Http;
using Newtonsoft.Json;

namespace Chapter04.Examples
{
    public static class DynamicJson
    {
        public static string WhatToDo()
        {
            const string fetchUrl = "https://jsonplaceholder.typicode.com/todos";
            using var client = new WebClient();

            var json = client.DownloadString(fetchUrl);

            // instead of creating objects, can just use dynamics
            dynamic objects = JsonConvert.DeserializeObject(json);

            // json looks like this
            //{
            //  "userId": 1,
            //  "id": 1,
            //  "title": "delectus aut autem",
            //  "completed": false
            // },...

            foreach (dynamic todo in objects)
            {
                if (todo.completed == false) 
                    return todo.title;
            }

            return "Nothing!";
        }

        public static void MakeAPost()
        {
            const string postUrl = "https://jsonplaceholder.typicode.com/posts";
            var client = new WebClient();

            // need to send
            // {
            //  title: 'foo',
            //  body: 'bar',
            //  userId: 1
            // }
        
            dynamic post = new ExpandoObject();

            post.title = "Today I Learned";
            post.body = "I learned stuff about dynamics, unfortunately...";
            post.userId = 42069;

            var serialized = JsonConvert.SerializeObject(post);
            Console.WriteLine($"Uploading post... {serialized}");

            var response = client.UploadString(postUrl, serialized);

            // can break down the response as well,
            // which returns json with a property "id" of the new post
            dynamic obj = JsonConvert.DeserializeObject(response);

            Console.WriteLine($"PostId: {obj.id}");
        }
    }

}