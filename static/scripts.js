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

    function loadImgs() {
        const data = $("#sendData").serialize();

        resultImg.addClass('is-hidden');
        errorImg.addClass('is-hidden');
        loading.removeClass('is-hidden');

        $.ajax({
            url: "/run_new",
            data: data,
            method: 'POST',
            success: function(response) {
                loading.addClass('is-hidden');
                $("#resultImg1").attr('src', `data:image/png;base64,${response.img1}`);
                $("#resultImg2").attr('src', `data:image/png;base64,${response.img2}`);
                resultImg.removeClass('is-hidden');

            },
            error: function(response) {
                loading.addClass('is-hidden');
                errorImg.removeClass('is-hidden');
                console.log(response);
            }
        })
    }

    $("#sumbitForm").on('click', function(e) {
        e.preventDefault();
        loadImgs();
    })

    // Load 0 img when form data change
    form.on('change', function() {
        load0Img();
    })

    // Load 0 img on load page
    load0Img();

});
