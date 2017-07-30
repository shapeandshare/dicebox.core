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
            console.log(json_out);

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

/*    function rgb2hsl(r, g, b) {
        r /= 255; g /= 255; b /= 255;

        var min = Math.min(r, g, b);
        var max = Math.max(r, g, b);
        var delta = max - min;
        var h, s, l;

        if (max == min) {
            h = 0;
        } else if (r == max) {
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else if (b == max) {
            h = 4 + (r - g) / delta;
        }

        h = Math.min(h * 60, 360);

        if (h < 0) {
            h += 360;
        }

        l = (min + max) / 2;

        if (max == min) {
            s = 0;
        } else if (l <= 0.5) {
            s = delta / (max + min);
        } else {
            s = delta / (2 - max - min);
        }

        return [h, s * 100, l * 100];
    }*/

    addEventListener("DOMContentLoaded", initialize);
})(document, navigator);
