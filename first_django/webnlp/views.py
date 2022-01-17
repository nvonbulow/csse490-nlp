from django.shortcuts import render
import re

# Create your views here.
def home(request):
    return render(request, 'home.html', {})

def regex(request):
    if request.method == 'POST':
        phonepattern = r'\d{3}-\d{3}-\d{4}'
        text = request.POST['text']
        phone = re.findall(phonepattern, text)
        return render(request, 'regex.html', { 'phone': phone })
    return render(request, 'regex.html', {})

def lemma(request):
    return render(request, 'lemma.html', {})

def pos(request):
    return render(request, 'pos.html', {})

def ner(request):
    return render(request, 'ner.html', {})
