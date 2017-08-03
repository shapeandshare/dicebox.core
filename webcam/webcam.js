(function (doc, nav) {
    "use strict";

    var video, width, height, context;
    var encoded_data;

    function initialize() {
        // The source video.
        video = doc.getElementById("video");
        width = video.width;
        height = video.height;

        // The target canvas.
        var canvas = doc.getElementById("canvas");
        context = canvas.getContext("2d");

        // Get the webcam's stream.
        nav.getUserMedia({video: true}, startStream, function () {});
    }

    function startStream(stream) {
        video.src = URL.createObjectURL(stream);
        video.play();

        // Ready! Let's start drawing.
        requestAnimationFrame(draw);
    }

    function draw() {
        var frame = readFrame();

        if (frame) {
            // perform transform on input data
            // e.g. make gray scale
            makeGrayScale(frame.data)
            context.putImageData(frame, 0, 0);

            var dataURL = canvas.toDataURL();
            encoded_data = dataURL.replace(/^data:image\/png;base64\,/, "")
            //console.log(encoded_data);

            var json_packet={"data":encoded_data};
            var json_out = JSON.stringify(json_packet);
            //console.log(json_out);

            $.ajax({
                url: 'http://localhost:5000/api/classify',
                type: 'POST',
                headers:
                    {
                        'Content-type': 'application/json',
                        'API-ACCESS-KEY': '6e249b5f-b483-4e0d-b50b-81d95e3d9a59',
                        'API-VERSION': '0.2.2'
                    },
                data: json_out,
                success: function(html) {
                    console.log(html);
                    //$("#results").append(html);
                }
            });

        }

        // Wait for the next frame.
        requestAnimationFrame(draw);
    }

    function readFrame() {
        try {
            var sourceX = width/2 - 50;
            var sourceY = height/2 - 50;

            //var sourceWidth = width;
            //var sourceHeight = height;
            var sourceWidth = 100;
            var sourceHeight = 100;

            var destWidth = 100;
            var destHeight = 100;
            //var destX = width / 2 - destWidth / 2;
            //var destY = height / 2 - destHeight / 2;
            var destX = 0;
            var destY = 0;
            context.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, destX, destY, destWidth, destHeight);
            //context.drawImage(video, width/2 - 50, height/2 - 50, width/2 + 50, height/2 + 50, 0, 0, 100, 100);
        } catch (e) {
            // The video may not be ready, yet.
            return null;
        }
        return context.getImageData(0, 0, destWidth, destHeight);
    }

    function makeGrayScale(data) {
        var len = data.length;
        var luma;

        for (var i = 0; i < len; i += 4) {
            luma = data[i] * 0.2126 + data[i+1] * 0.7152 + data[i+2] * 0.0722;
            data[i] = data[i+1] = data[i+2] = luma;
            //data[i+3] = data[i+3];
//            // Convert from RGB to HSL...
//            var hsl = rgb2hsl(data[j], data[j + 1], data[j + 2]);
//            var h = hsl[0], s = hsl[1], l = hsl[2];
//
//            // ... and check if we have a somewhat green pixel.
//            if (h >= 90 && h <= 160 && s >= 25 && s <= 90 && l >= 20 && l <= 75) {
//                data[j + 3] = 0;
//            }
        }
    }

    addEventListener("DOMContentLoaded", initialize);
})(document, navigator);
