let pulse;
loadPulse()
    .then(module => {
        pulse = module;
    }).catch(err => {
        console.log("Error in loading module: ", err);
    });

/**
 * @param {number} N - FFT points (must be a power of 2, and at least 2)
 * @returns {(x: Float32Array | Float64Array) => Float64Array} FFT function.
 *   must be passed an array of length 2*N (real0, imag0, real1, imag1, ...).
 *   the argument will be left untouched and an array of the same length and layout
 *   will be returned with the result. the first complex number of the array is DC.
 *   **the buffer is reused/overwritten on future invocations.**
 */
function makeFFT(N) {
	if (N !== (N >>> 0))
		throw new Error('not an u32')
	const Nb = Math.log2(N) | 0
	if (N !== (1 << Nb) || Nb < 1)
		throw new Error('not a power of two, or too small')

	// calculate the twiddle factors
	const twiddle = new Float64Array(2*N)
	for (let i = 0; i < N; i++) {
		const arg = - 2 * Math.PI * i / N;
		twiddle[2*i+0] = Math.cos(arg);
		twiddle[2*i+1] = Math.sin(arg);
	}

	const sizeMask = 2*N - 1;
	function phase(
		/** @type {number} */ idx,
		/** @type {Float64Array | Float32Array} */ src,
		/** @type {Float64Array | Float32Array} */ dst,
	) {
		const stride = 1 << (Nb - idx), mask = stride - 1
		for (let o = 0; o < 2*N; o += 2) {
			const i = o & ~mask, O = o & mask
			const i1 = ((i<<1) & sizeMask) + O, i2 = i1 + stride
			// if JS had complexes: dst[o] = src[i1] + twiddle[i] * src[i2]
			dst[o+0] = src[i1+0] + twiddle[i+0] * src[i2+0] - twiddle[i+1] * src[i2+1];
			dst[o+1] = src[i1+1] + twiddle[i+0] * src[i2+1] + twiddle[i+1] * src[i2+0];
		}
	}

	let a = new Float64Array(2*N)
	let b = new Float64Array(2*N)
	return function fft(src) {
		if (src.length !== 2*N)
			throw new Error('invalid length')
		phase(0, src, a)
		for (let idx = 1; idx < Nb; idx++) {
			phase(idx, a, b)
			; [a, b] = [b, a]
		}
		return a
	}
}

/* Utility functions to generate arbitrary input in various formats */
function inputReals(size) {
    var result = new Float32Array(size);
    for (var i = 0; i < result.length; i++)
	result[i] = (Math.random()*2-1) / 4.0;
    return result;
}

function inputInterleaved(size) {
    var result = new Float32Array(size*2);
    for (var i = 0; i < size; i++)
	result[i*2] = (Math.random()*2-1) / 4.0;
    return result;
}

var iterations = 2000;

function report(name, start, middle, end, size) {
    function addTo(tag, thing) {
	    document.getElementById(name + "-" + tag).innerHTML += thing + "<br>";
    }
    addTo("size", size);
    addTo("1", Math.round(middle - start) + " ms");
    addTo("2", Math.round(end - middle) + " ms");
    addTo("itr", Math.round((1000.0 /
			     ((end - middle) / iterations))) + " itr/sec");
}

function testFFTasm(size) {
    var fft = new KissFFTR(size);
    var ri = [...Array(64)].map(() => inputReals(size));

    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
	if (i == iterations) {
	    middle = performance.now();
	}
    var out = fft.forward(ri[i%ri.length]);
    }

    var end = performance.now();

    report("kissfft", start, middle, end, size);

    fft.dispose();
}

function testFFTCCasm(size) {
    var fft = new KissFFT(size);
    var cin = [...Array(64)].map(() => inputInterleaved(size));

    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
	if (i == iterations) {
	    middle = performance.now();
	}
    var out = fft.forward(cin[i%cin.length]);
    }

    var end = performance.now();

    report("kissfftcc", start, middle, end, size);

    fft.dispose();
}

function testFFTwasm(size) {
    var fft = new pulse.fftReal(size);
    var ri = [...Array(64)].map(() => inputReals(size));
  
    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
      if (i == iterations) {
        middle = performance.now();
      }
      var out = fft.forward(ri[i%ri.length]);
    }
    var end = performance.now();

    report("WASMkissfft", start, middle, end, size);

    fft.dispose();
  }

  function testFFTCCwasm(size) {
    var fft = new pulse.fftComplex(size);
    var cin = [...Array(64)].map(() => inputInterleaved(size));

    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
      if (i == iterations) {
        middle = performance.now();
      }
      var out = fft.forward(cin[i%cin.length]);
    }

    var end = performance.now();

    report("WASMkissfftcc", start, middle, end, size);

    fft.dispose();
  }

  function testFFTCCjs(size) {
    var fft = makeFFT(size);
    var cin = [...Array(64)].map(() => inputInterleaved(size));

    var start = performance.now();
    var middle = start;
    var end = start;

    total = 0.0;

    for (var i = 0; i < 2*iterations; ++i) {
      if (i == iterations) {
        middle = performance.now();
      }
      var out = fft(cin[i%cin.length]);
    }

    var end = performance.now();

    report("JSfftcc", start, middle, end, size);
  }

var sizes = [ 4, 8, 512, 2048, 4096, 8192 ];
var tests = [testFFTasm, testFFTCCasm, testFFTwasm, testFFTCCwasm, testFFTCCjs];
var nextTest = 0;
var nextSize = 0;
var interval;

function test() {
    clearInterval(interval);
    if (nextTest == tests.length) {
        nextSize++;
        nextTest = 0;
        if (nextSize == sizes.length) {
            return;
        }
    }
    f = tests[nextTest];
    size = sizes[nextSize];
    nextTest++;
    f(size);
    interval = setInterval(test, 200);
}

window.onload = function() {
    document.getElementById("test-description").innerHTML =
    "The table above compares two fft functions from the kissfft library compiled to both assembly and WebAssembly. 4000 iterations of each algorithm were performed on sample data of several different sizes (4, 8, 512, 2048, and  4096).<br> The times for the first and second half of each test are shown separately to elucidate any differences caused by the Javascript engine warming up. The last column shows the rate at which each algorithm performed the 4000 iterations. For the larger buffer sizes, the Web Assembly compiled functions are consistently faster than their assembly copiled counterparts."
	// "Running " + 2*iterations + " iterations per implementation.<br>Timings are given separately for the first half of the run (" + iterations + " iterations) and the second half, in case the JS engine takes some warming up.<br>Each cell contains results for the following buffer sizes: " + sizes[0] + ', ' + sizes[1] + ', ' + sizes[2] + ', ' + sizes[3] + ', ' + sizes[4] + '.';
    interval = setInterval(test, 200);
}

function testWASMmic (size, freqData) {
    var magnitudes = new Float32Array((size / 2) + 1);
    var fft = new pulse.fftReal(size);

    var ri = freqData;
    var out = fft.forward(ri);

    for (var j = 0; j <= size/2; ++j) {
      magnitudes[j] = Math.sqrt(out[j*2] * out[j*2] + out[j*2+1] * out[j*2+1]);
    }

    fft.dispose();
    return magnitudes;
  }



  var webaudio_tooling_obj = function () {
    var frequencyData = new Float32Array(1025);
    var svgHeight = '600';
    var svgWidth = '1200';
    var barPadding = '1';

    function createSvg(parent, height, width) {
      return d3.select(parent).append('svg').attr('height', height).attr('width', width);
    }
    var svg = createSvg('#graph', svgHeight, svgWidth);

    // Set up initial D3 chart.
    svg.selectAll('rect')
    .data(frequencyData)
    .enter()
    .append('rect')
    .attr('x', function (d, i) {
      return i * (svgWidth / frequencyData.length);
    })
    .attr('width', svgWidth / frequencyData.length);



    var audioContext = new AudioContext(),
        microphone_stream = null,
        script_processor_node = null;


    if (!navigator.getUserMedia)
            navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia ||
                          navigator.mozGetUserMedia || navigator.msGetUserMedia;
    if (navigator.getUserMedia){
        navigator.getUserMedia({audio:true},
          function(stream) {
              start_microphone(stream);
          },
          function(e) {
            console.log('Error capturing audio: ', e);
          }
        );
    } else { alert('getUserMedia not supported in this browser.'); }




    function start_microphone(stream){
      microphone_stream = audioContext.createMediaStreamSource(stream);
      script_processor_node = audioContext.createScriptProcessor(2048, 1, 1);

      microphone_stream.connect(script_processor_node);
      script_processor_node.connect(audioContext.destination);

      script_processor_node.onaudioprocess = process_microphone_buffer;


      function process_microphone_buffer(event) {
          var microphone_input_buffer = event.inputBuffer.getChannelData(0);

          function hanningWindow(inputBuffer) {
            for (var i = 1; i < inputBuffer.length; i += 1) {
              inputBuffer[i] *= 0.54 - ((0.46) * (Math.cos((2 * Math.PI * i)/(microphone_input_buffer.length - 1))))
            }
            return inputBuffer;
          }

          var windowedInput = hanningWindow(microphone_input_buffer);
          frequencyData = testWASMmic(windowedInput.length, windowedInput);
      }

      function renderChart() {
        requestAnimationFrame(renderChart);
        svg.selectAll('rect')
        .data(frequencyData)
        .attr('y', function(d) {
          return .5 *(svgHeight - (d * 25));
        })
        .attr('height', function(d) {
          return d * 25;
        })
        .attr('fill', function(d) {
          return '#adc5ff';
        });
      }
      renderChart();
    }
  }();
