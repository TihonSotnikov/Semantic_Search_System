document.getElementById('search-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const query = document.getElementById('search-input').value;
    if (query.trim() === '') return;

    fetch('/search?text=' + encodeURIComponent(query))
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            data.forEach(item => {
                const div = document.createElement('div');
                div.innerHTML = `<p>${item[0]}</p><p><small>${item[1]}</small></p>`;
                div.classList.add('result-item');
                resultsDiv.appendChild(div);
            });
        })
        .catch(error => console.error('Error:', error));
});
