#include <iostream>
#include <vector>
#include <random>
#include <string>

using namespace std;

class Minesweeper {
private:
    int rows, cols, mines;
    vector<vector<int>> board;
    vector<vector<bool>> revealed;
    vector<vector<bool>> flagged;
    bool gameOver;
    bool gameWon;
    bool firstMove;
    int revealedCount;

    void initBoard() {
        board.assign(rows, vector<int>(cols, 0));
        revealed.assign(rows, vector<bool>(cols, false));
        flagged.assign(rows, vector<bool>(cols, false));
        gameOver = false;
        gameWon = false;
        firstMove = true;
        revealedCount = 0;
    }

    void placeMines() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(0, rows * cols - 1);

        int placed = 0;
        while (placed < mines) {
            int pos = dist(gen);
            int r = pos / cols;
            int c = pos % cols;
            if (board[r][c] != -1) {
                board[r][c] = -1;
                placed++;
            }
        }
        calculateNumbers();
    }

    void calculateNumbers() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (board[i][j] != -1)
                    board[i][j] = 0;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (board[i][j] == -1)
                    updateNumbers(i, j);
    }

    void updateNumbers(int r, int c) {
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && board[nr][nc] != -1) {
                    board[nr][nc]++;
                }
            }
        }
    }


    void ensureSafeFirstMove(int r, int c) {
        if (board[r][c] != -1)
            return;

        board[r][c] = 0;
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(0, rows * cols - 1);

        while (true) {
            int pos = dist(gen);
            int nr = pos / cols;
            int nc = pos % cols;
            if (board[nr][nc] != -1 && !(nr == r && nc == c)) {
                board[nr][nc] = -1;
                break;
            }
        }
        calculateNumbers();
    }

    void reveal(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || revealed[r][c] || flagged[r][c])
            return;

        revealed[r][c] = true;
        revealedCount++;

        if (board[r][c] == 0) {
            for (int dr = -1; dr <= 1; dr++)
                for (int dc = -1; dc <= 1; dc++)
                    reveal(r + dr, c + dc);
        }
    }

public:
    Minesweeper(int r, int c, int m) : rows(r), cols(c), mines(m) {
        initBoard();
        placeMines();
    }

    void printBoard() {
        cout << "\n  ";
        for (int j = 0; j < cols; j++)
            cout << j << " ";
        cout << "\n";

        for (int i = 0; i < rows; i++) {
            cout << i << " ";
            for (int j = 0; j < cols; j++) {
                if (gameOver && board[i][j] == -1)
                    cout << "* ";
                else if (flagged[i][j])
                    cout << "F ";
                else if (revealed[i][j]) {
                    if (board[i][j] == 0)
                        cout << ". ";
                    else
                        cout << board[i][j] << " ";
                } else
                    cout << "# ";
            }
            cout << "\n";
        }
        cout << "\nMines: " << mines << "  Flags: " << countFlags() << "  Remaining: " << mines - countFlags() << "\n";
    }

    int countFlags() {
        int count = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (flagged[i][j])
                    count++;
        return count;
    }

    bool isGameOver() { return gameOver; }
    bool isGameWon() { return gameWon; }

    void toggleFlag(int r, int c) {
        if (r >= 0 && r < rows && c >= 0 && c < cols && !revealed[r][c] && !gameOver)
            flagged[r][c] = !flagged[r][c];
    }

    bool revealCell(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || revealed[r][c] || flagged[r][c] || gameOver)
            return true;

        if (firstMove) {
            ensureSafeFirstMove(r, c);
            firstMove = false;
        }

        if (board[r][c] == -1) {
            gameOver = true;
            return false;
        }

        reveal(r, c);

        if (revealedCount == rows * cols - mines) {
            gameWon = true;
            gameOver = true;
        }
        return true;
    }

    void revealAll() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                revealed[i][j] = true;
        gameOver = true;
    }
};

int main() {
    cout << "=== MINESWEEPER ===\n\n";
    cout << "Select difficulty:\n";
    cout << "1. Easy   (9x9, 10 mines)\n";
    cout << "2. Medium (16x16, 40 mines)\n";
    cout << "3. Hard   (30x16, 99 mines)\n";
    cout << "4. Custom\n\n";
    cout << "Choice: ";

    int choice;
    cin >> choice;

    int rows, cols, mines;
    switch (choice) {
        case 1: rows = 9; cols = 9; mines = 10; break;
        case 2: rows = 16; cols = 16; mines = 40; break;
        case 3: rows = 16; cols = 30; mines = 99; break;
        default:
            cout << "Rows: "; cin >> rows;
            cout << "Cols: "; cin >> cols;
            cout << "Mines: "; cin >> mines;
    }

    if (rows <= 0 || cols <= 0 || mines < 0 || mines >= rows * cols) {
        cout << "Invalid board size or mine count!\n";
        return 1;
    }

    Minesweeper game(rows, cols, mines);
    game.printBoard();

    cout << "\nCommands: r row col (reveal), f row col (flag), q (quit)\n";

    while (!game.isGameOver()) {
        char cmd;
        int r, c;
        cout << "\n> ";
        cin >> cmd;

        if (cmd == 'q') break;

        cin >> r >> c;
        if (cmd == 'r') {
            if (!game.revealCell(r, c)) {
                game.revealAll();
                game.printBoard();
                cout << "\n*** BOOM! GAME OVER ***\n";
                break;
            }
        } else if (cmd == 'f') {
            game.toggleFlag(r, c);
        }

        game.printBoard();

        if (game.isGameWon()) {
            cout << "\n*** CONGRATULATIONS! YOU WIN! ***\n";
            break;
        }
    }

    return 0;
}
