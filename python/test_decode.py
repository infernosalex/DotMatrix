import random
import string
from qr_decode import QRDecode
from qr_gen import QRCode


def test_1():
    for i in range(1, 4):
        random_text = ''.join(random.choices(string.ascii_letters + string.punctuation + string.digits, k=100))
        qr = QRCode(random_text, version=-1, error_correction="L", mask=-1, debug=False, mode="auto")
        matrix = qr.get_matrix()

        qrd = QRDecode(matrix, debug=True)
        message = qrd.decode()
        print(f"[TEST] Decoded message: {message}")
        assert message == random_text, f"Failed to decode message: {random_text}, got {message}"


def test_2():
    for i in range(1, 4):
        random_text = ''.join(random.choices(string.ascii_letters + string.punctuation + string.digits, k=100))
        qr = QRCode(random_text, version=-1, error_correction="L", mask=-1, debug=False, mode="auto")
        matrix = qr.get_matrix()

        qrd = QRDecode(matrix, debug=True)
        message = qrd.decode()
        print(f"[TEST] Decoded message: {message}")
        assert message == random_text, f"Failed to decode message: {random_text}, got {message}"

def test_3():
    for i in range(1, 4):
        random_text = ''.join(random.choices(string.ascii_letters + string.punctuation + string.digits, k=100))

        qr = QRCode(random_text, version=-1, error_correction="M", mask=-1, debug=False, mode="auto")
        matrix = qr.get_matrix()

        qrd = QRDecode(matrix, debug=True)
        message = qrd.decode()
        print(f"[TEST] Decoded message: {message}")
        assert message == random_text, f"Failed to decode message: {random_text}, got {message}"

def test_4():
    for i in range(1, 4):
        random_text = ''.join(random.choices(string.ascii_letters + string.punctuation + string.digits, k=100))
        qr = QRCode(random_text, version=-1, error_correction="H", mask=-1, debug=False, mode="auto")
        matrix = qr.get_matrix()

        qrd = QRDecode(matrix, debug=True)
        message = qrd.decode()
        print(f"[TEST] Decoded message: {message}")
        assert message == random_text, f"Failed to decode message: {random_text}, got {message}"

def find_breaking_point():
    for i in range(1, 300):
        for ecc in ["L", "M", "Q", "H"]:
            random_text = ''.join(random.choices(string.ascii_letters + string.punctuation + string.digits, k=i))
            qr = QRCode(random_text, version=-1, error_correction=ecc, mask=-1, debug=False, mode="auto")
            matrix = qr.get_matrix()
            qrd = QRDecode(matrix, debug=True)

            message = qrd.decode()
            if message != random_text:
                print(f"[TEST] Breaking point found at {i} characters with {ecc} error correction")
                assert False, f"Failed to decode message: {random_text}, got {message}"
                break

def known_fail():
    qr_code = QRCode('OkTuN6f3ue)c)Qu<P^AACk}j=&=_j9yn~3A(F-G42ti^rVPkgji;mA\'K~zigaBr*#\\q}S2K-I-"G/-', version=-1, error_correction='M', mask=-1, debug=False, mode="byte")
    matrix = qr_code.get_matrix()
    qrd = QRDecode(matrix, debug=True)
    message = qrd.decode()
    print(f"[TEST] Decoded message: {message}")
    assert message == 'OkTuN6f3ue)c)Qu<P^AACk}j=&=_j9yn~3A(F-G42ti^rVPkgji;mA\'K~zigaBr*#\\q}S2K-I-"G/-', f'Failed to decode message: OkTuN6f3ue)c)Qu<P^AACk}}j=&=_j9yn~3A(F-G42ti^rVPkgji;mA\'K~zigaBr*#\\q}}S2K-I-"G/-, got {message}'



if __name__ == '__main__':
    #test_1() # OK
    #test_2() # OK
    #test_3()
    #test_4()
    #find_breaking_point()
    known_fail()
    print("[TEST] All tests passed.")

