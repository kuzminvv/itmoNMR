$(document).ready(function() {

    const form = $("#sendData");
    const loading = $("#loading");
    const resultImg = $("#resultImg");
    const errorImg = $("#errorMessage");

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
        resultImg.addClass('is-hidden');
        errorImg.addClass('is-hidden');
        loading.removeClass('is-hidden');
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
        const data = $("#sendData").serialize();
        $("#resultImg1").attr('src', `data:image/png;base64,${response.img1}`);
        $("#resultImg2").attr('src', `data:image/png;base64,${response.img2}`);
        loading.addClass('is-hidden');
        resultImg.removeClass('is-hidden');
    })

    // Load 0 img when form data change
    form.on('change', function() {
        load0Img();
    })

    // Load 0 img on load page
    load0Img();

});
