window.onload = () => {

    // DOM Elements
    const $submit = document.querySelector('.js-submit');
    const $button = document.querySelector('.js-button');
    const $loader = document.querySelector('.js-loader');
    const $result = document.querySelector('.js-result');
    const $error = document.querySelector('.js-error');

    const $object = document.querySelector('input[type=radio][name=object]:checked');
    const $gradient = document.querySelector('input[type=radio][name=gradient]:checked');

    // Engine functions

    const glassOfWaterPreset = [10, 20, 30];
    function fft () {

    }
    function imageGenerate(){

    }

    async function calculate(){

        console.log('object', $object.value);
        console.log('gradient', $gradient.value);

        var test;
        test = glassOfWaterPreset[1] + glassOfWaterPreset[2] + glassOfWaterPreset[0];

        //преобразование фурье
        fft();

        //функция которая генерит изображение результата
        imageGenerate ();

        $button.classList.add('is-hidden')
        $error.classList.add('is-hidden')
        $result.classList.add('is-hidden')
        $loader.classList.remove('is-hidden')

        fetch("https://reqbin.com/echo/post/json", {
            method: "POST",
            body: {
                object: $object.value,
                    gradient: $gradient.value
            }
        }).then(
            response => response.json()
        ).then(
            json => {
                console.log(json);
                $button.classList.remove('is-hidden')
                $result.classList.remove('is-hidden')
                $loader.classList.add('is-hidden')
            }
        ).catch(e => {
            $button.classList.remove('is-hidden')
            $error.classList.remove('is-hidden')
            $loader.classList.add('is-hidden')
        });
    }

    // DOM event functions
    $submit.addEventListener('click', () => {
        calculate();
    });
}