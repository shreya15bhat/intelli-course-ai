const menuToggle=document.getElementById('menuToggle');if(menuToggle)menuToggle.addEventListener('click',()=>document.getElementById('sidebar').classList.toggle('open'));
document.addEventListener('click',e=>{const s=document.getElementById('sidebar');if(s&&s.classList.contains('open')&&!s.contains(e.target)&&!e.target.closest('#menuToggle'))s.classList.remove('open')});
