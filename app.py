import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM



menu = st.sidebar.selectbox(
    'Оберіть сторінку', 
    ['Приклад використання', 'Генерація опису товару']
)

if menu == 'Приклад використання':
  st.title('Приклад роботи моделі, що генерує опис товарів на основі назви та характеристик.')
  st.write('Приклад введених даних:')
  st.code("""
      Назва: 2  Гель для прання Polar Shine Marseille з марсельським милом, 125 циклів прання, 5 л;
      Бренд: Polar Shine;
      Ключові інгредієнти: 5-15% аніонні ПАР, &lt 5% хлорид натрію, запашник, піногасник, метилхлорізотіазолінон та метилізотіазолінон, барвник.
      """, language='text')
  st.write("Приклад роботи моделі:")
  st.code("""
  Назва: Гель для прання Polar Shine Marseille з марсельським милом, 125 циклів прання, 5 л;
      Бренд: Polar Shine;
      Ключові інгредієнти: 5-15% аніонні ПАР, &lt 5% хлорид натрію, запашник, піногасник, метилхлорізотіазолінон та метилізотіазолінон, барвник.
      Опис:ЗВЕРНІТЬ УВАГУ! Товар має пошкоджене паковання.Ефективне очищення та дбайливий догляд за тканинними виробами
      завдяки унікальному поєднанню універсального порошку з мінеральним пінним активатором у ньому.
      Переваги продукту: Видаляє забруднення різної природи та стійкості.
      Зберігає цілісність волокна, запобігаючи його деформації та передчасному зношуванню.
      Дарує речам приємний квітковий аромат.Як діє?
      Активні компоненти засобу помякшують воду, утворюючи мяку піну, яка швидко розчиняється у воді та повноцінно усуває навіть найдрібніші плями,
      зокрема ті, які викликають небажаний запах.
      """, language='text')
  
elif menu=='Генерація опису товару':
  st.title('Напишіть назву товару з його характеристиками, для якого хочете створити опис.')
  st.write("Введіть деталі продукту:")

  product_name = st.text_input('Назва продукту')
  brand_name = st.text_input('Бренд')
  ingredients = st.text_area('Ключові інгредієнти')


  tokenizer = AutoTokenizer.from_pretrained("Alinkaaa1/Llama-2-7b_ukr_item_descr", trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")#"Alinkaaa1/Llama-2-7b_ukr_item_descr", trust_remote_code=True)
  if st.button('Передати дані до моделі'):
      if product_name and brand_name and ingredients:
          input_string = f"Назва: {product_name}; Бренд: {brand_name}; Ключові інгредієнти: {ingredients}"
          st.write("Дані передано до моделі:")
          st.write(input_string)

          outputs = model.generate(input_string['input_ids'],
                max_length=450,
                num_beams=5,
                early_stopping=False,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
          generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
          st.write(f"Результат моделі: {generated_text}")
      else:
          st.warning("Будь ласка, заповніть всі поля.")
