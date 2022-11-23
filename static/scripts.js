$(document).ready(function() {

    const form = $("#sendData");
    const loading = $("#loading");
    const resultImg = $("#resultImg");
    const errorImg = $("#errorMessage");

    function drawcanvas() {
        const start = new Date().getTime();
        var canvas = document.getElementById("canvasimg");
        if (canvas.getContext) {
            var ctx = canvas.getContext("2d");
            var myImageData = ctx.createImageData(280, 280);
            var colarr = myImageData.data;
            for (var i = 0; i < 280; i++) {
                for (var j = 0; j < 300; j++) {
                    colarr[(i * 280 + j) * 4] = Math.floor(255-0.91*i);
                    colarr[(i * 280 + j) * 4 + 1] = Math.floor(255-0.91*j);
                    colarr[(i * 280 + j) * 4 + 2] = 0;
                    colarr[(i * 280 + j) * 4 + 3] = 255;
                }
            }
            ctx.putImageData(myImageData, 0, 0)
        }
        const end = new Date().getTime();
        $(".title").text(`${end-start}`)
    }


    function dataToRGBA(array){
        var maxVal = Math.max.apply(null, array);
        var colArray = new Uint8ClampedArray(array.length*4);
        for (let i = 0; i < array.length; i++ ) {
            colArray[4 * i] = Math.floor(array[i] / maxVal * 255);
            colArray[4 * i + 1] = 0;
            colArray[4 * i + 2] = 0;
            colArray[4 * i + 3] = 250;
        }
        return colArray;
    }

    function kToRGBA(array){
        var maxVal = Math.max.apply(null, array);
        var colArray = new Uint8ClampedArray(array.length*4);
        for (let i = 0; i < array.length; i++ ) {
            colArray[4 * i] = Math.floor(array[i] / maxVal * 255);
            colArray[4 * i + 1] = 0;
            colArray[4 * i + 2] = 0;
            colArray[4 * i + 3] = 250;
        }
        return colArray;
    }


    function draw_array(colorData, cvs_obj) {
//        var canvas = document.getElementById("canvasimg");
        if (cvs_obj.getContext) {
            var ctx = cvs_obj.getContext("2d");
            var cd = dataToRGBA(colorData);
            var myImageData = ctx.createImageData(280, 280, cd);
            var colarr = myImageData.data;
            for (var i = 0; i < cd.length; i++) {
                colarr[i] = cd[i];
            }
            ctx.putImageData(myImageData, 0, 0);
        }
    }

    function load0Img() {
        const data = $("#sendData").serialize();
        errorImg.addClass('is-hidden');
        $('#loading0').removeClass('is-hidden');
        $("#resultImg0").addClass('is-hidden');
        $.ajax({
            url: "/run_0_img",
            data: data,
            method: 'POST',
            success: function(response) {
                $("#resultImg0").removeClass('is-hidden')
                $("#resultImg0").attr('src', `data:image/png;base64,${response.img0}`);

            },
            error: function(response) {
                errorImg.removeClass('is-hidden');
                console.log(response);
            },
            complete: function() {
                $('#loading0').addClass('is-hidden');
            }
        })
    }

    $("#sumbitForm").on('click', function(e) {
        e.preventDefault();
//        resultImg.addClass('is-hidden');
        errorImg.addClass('is-hidden');
//        loading.removeClass('is-hidden');
        socket.emit('run_new', getFormData(form));
    })

    function getFormData($form){
        var unindexed_array = $form.serializeArray();
        var indexed_array = {};

        $.map(unindexed_array, function(n, i){
            indexed_array[n['name']] = n['value'];
        });

        return indexed_array;
    }

    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('new_res', function(response) {
        $(".title").text('Working...')
        const data = $("#sendData").serialize();
        $("#resultImg1").attr('src', `data:image/png;base64,${response.img1}`);
        $("#resultImg2").attr('src', `data:image/png;base64,${response.img2}`);
        loading.addClass('is-hidden');
        resultImg.removeClass('is-hidden');
    })

    socket.on('finish', function(response) {
        $(".title").text('Done!')
    })


    socket.on('newnewres', function(response) {
//        const start = new Date().getTime();
//        const end = new Date().getTime();
//        $(".title").text(`${end-start}`)
        var mri = JSON.parse(response[0]);
        var mri_cvs = document.getElementById("mri_cvs");
        var k_space = JSON.parse(response[1]);
        var k_space_cvs = document.getElementById("k_space_cvs");
        draw_array(mri, mri_cvs);
        draw_array(k_space, k_space_cvs);
        resultImg.removeClass('is-hidden');
    })

    // Load 0 img when form data change
    form.on('change', function() {
        load0Img();
    })

    drawcanvas();
    // Load 0 img on load page
    load0Img();

});
